# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
FastAPI routes for Kiro Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    APP_VERSION,
)
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_openai import build_kiro_payload
from kiro.streaming_openai import stream_kiro_to_openai, collect_stream_response, stream_with_first_token_retry
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security scheme ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.
    
    Expects format: "Bearer {PROXY_API_KEY}"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Kiro Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health():
    """
    Detailed health check.
    
    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION
    }

@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    """
    Return list of available models.
    
    Models are loaded at startup (blocking) and cached.
    This endpoint returns the cached list.
    
    Args:
        request: FastAPI Request for accessing app.state
    
    Returns:
        ModelList with available models in consistent format (with dots)
    """
    logger.info("Request to /v1/models")
    
    # Get available models based on mode
    if request.app.state.account_system:
        # Account system: collect models from all initialized accounts
        available_model_ids = request.app.state.account_manager.get_all_available_models()
    else:
        # Legacy: use resolver from first account
        account = request.app.state.account_manager.get_first_account()
        available_model_ids = account.model_resolver.get_available_models()
    
    # Build OpenAI-compatible model list
    openai_models = [
        OpenAIModel(
            id=model_id,
            owned_by="anthropic",
            description="Claude model via Kiro API"
        )
        for model_id in available_model_ids
    ]
    
    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.
    
    Accepts requests in OpenAI format and translates them to Kiro API.
    Supports streaming and non-streaming modes.
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in OpenAI ChatCompletionRequest format
    
    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)
    
    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.models_openai import ChatMessage
    
    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0
    
    for msg in request_data.messages:
        # Check if this is a tool_result for a truncated tool call
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                # Modify tool_result content to include truncation notice
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info
                )
                # Prepend truncation notice to original content
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                
                # Create NEW ChatMessage object (Pydantic immutability)
                modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                tool_results_modified += 1
                logger.debug(f"Modified tool_result for {msg.tool_call_id} to include truncation notice")
                continue  # Skip normal append since we already added modified version
        
        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
            truncation_info = get_content_truncation(msg.content)
            if truncation_info:
                # Add this message first
                modified_messages.append(msg)
                # Then add synthetic user message about truncation
                synthetic_user_msg = ChatMessage(
                    role="user",
                    content=generate_truncation_user_message()
                )
                modified_messages.append(synthetic_user_msg)
                content_notices_added += 1
                logger.debug(f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})")
                continue  # Skip normal append since we already added it
        
        modified_messages.append(msg)
    
    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(f"Truncation recovery: modified {tool_results_modified} tool_result(s), added {content_notices_added} content notice(s)")
    
    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================
    
    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        
        # Check if web_search already exists
        has_ws = any(
            getattr(tool, "type", None) == "function" and
            getattr(getattr(tool, "function", None), "name", None) == "web_search"
            for tool in request_data.tools
        )
        
        if not has_ws:
            from kiro.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information. Use when you need up-to-date data from the internet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")
    
    # ==============================================================================
    # Account System: Account System Failover or Legacy Mode
    # ==============================================================================
    
    if request.app.state.account_system:
        # ==============================================================================
        # ACCOUNT SYSTEM ENABLED: Failover Loop
        # ==============================================================================
        from kiro.account_errors import classify_error, ErrorType
        
        account_manager = request.app.state.account_manager
        all_accounts = list(account_manager._accounts.keys())
        MAX_ATTEMPTS = len(all_accounts) * 2  # Full circle with margin
        
        last_error_message = None
        last_error_status = None
        tried_accounts = set()  # Track tried accounts in current failover loop
        
        for attempt in range(MAX_ATTEMPTS):
            # Get next available account (excluding already tried)
            account = await account_manager.get_next_account(
                request_data.model,
                exclude_accounts=tried_accounts
            )
            
            if account is None:
                # All accounts unavailable
                if len(all_accounts) == 1:
                    # Single account - return original error with original status code
                    raise HTTPException(
                        status_code=last_error_status or 503,
                        detail=last_error_message or "Account unavailable"
                    )
                else:
                    # Multiple accounts - generic error with context
                    detail = "No available accounts for this model."
                    if last_error_message:
                        detail += f" Last error: {last_error_message}"
                    raise HTTPException(status_code=503, detail=detail)
            
            # Mark account as tried in current failover loop
            tried_accounts.add(account.id)
            
            # Use objects from account
            auth_manager = account.auth_manager
            model_cache = account.model_cache
            model_resolver = account.model_resolver
            
            # Generate conversation ID
            conversation_id = generate_conversation_id()
            
            # Build payload for Kiro
            profile_arn_for_payload = ""
            if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
                profile_arn_for_payload = auth_manager.profile_arn
            
            try:
                kiro_payload = build_kiro_payload(
                    request_data,
                    conversation_id,
                    profile_arn_for_payload
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # Log Kiro payload
            try:
                kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode('utf-8')
                if debug_logger:
                    debug_logger.log_kiro_request_body(kiro_request_body)
            except Exception as e:
                logger.warning(f"Failed to log Kiro request: {e}")
            
            # Create HTTP client
            url = f"{auth_manager.api_host}/generateAssistantResponse"
            logger.debug(f"Kiro API URL: {url} (account: {account.id})")
            
            if request_data.stream:
                http_client = KiroHttpClient(auth_manager, shared_client=None)
            else:
                shared_client = request.app.state.http_client
                http_client = KiroHttpClient(auth_manager, shared_client=shared_client)
            
            try:
                # Prepare token-counting data (needed by both stream paths)
                messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
                tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None

                if request_data.stream:
                    # Streaming: do NOT pre-await the upstream request here.
                    # Pre-awaiting blocks byte-silent for the entire prefill (TTFT)
                    # window before StreamingResponse is returned, so large-context
                    # follow-ups (e.g. after a tool call) look like a dead stream to
                    # the client. Make the request INSIDE the generator where
                    # keep-alive SSE comments flow during prefill
                    # (see stream_with_first_token_retry in streaming_core.py).
                    # NOTE: cross-account failover is not available for streaming in
                    # this path; upstream errors surface inside the generator. This
                    # is intentional (single-account deployments are unaffected).
                    await account_manager.report_success(account.id, request_data.model)

                    async def stream_wrapper():
                        streaming_error = None
                        client_disconnected = False
                        try:
                            async def make_retry_request():
                                return await http_client.request_with_retry(
                                    "POST", url, kiro_payload, stream=True
                                )

                            async for chunk in stream_with_first_token_retry(
                                make_request=make_retry_request,
                                client=http_client.client,
                                model=request_data.model,
                                model_cache=model_cache,
                                auth_manager=auth_manager,
                                initial_response=None,
                                request_messages=messages_for_tokenizer,
                                request_tools=tools_for_tokenizer
                            ):
                                yield chunk
                        except GeneratorExit:
                            client_disconnected = True
                            logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                        except Exception as e:
                            streaming_error = e
                            # StreamingResponse already committed HTTP 200 (status +
                            # headers sent, keepalive comments may have flowed during
                            # prefill). Cannot raise an HTTP error now — that triggers
                            # Starlette's "response already started". Emit the error
                            # in-band as an SSE chunk + [DONE] and end cleanly.
                            try:
                                import time as _t
                                err_chunk = {
                                    "id": "error",
                                    "object": "chat.completion.chunk",
                                    "created": int(_t.time()),
                                    "model": request_data.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": f"\n[kiro-gateway error] {e}"},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                                yield "data: [DONE]\n\n"
                            except Exception:
                                pass
                            # Do NOT re-raise: response already started.
                            return
                        finally:
                            await http_client.close()
                            if streaming_error:
                                error_type = type(streaming_error).__name__
                                error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                                logger.error(f"HTTP 500 - POST /v1/chat/completions (streaming) - [{error_type}] {error_msg[:100]}")
                            elif client_disconnected:
                                logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - client disconnected")
                            else:
                                logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - completed")
                            if debug_logger:
                                if streaming_error:
                                    debug_logger.flush_on_error(500, str(streaming_error))
                                else:
                                    debug_logger.discard_buffers()

                    return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

                # Non-streaming: pre-await so we can classify errors and fail over.
                response = await http_client.request_with_retry(
                    "POST",
                    url,
                    kiro_payload,
                    stream=True
                )

                if response.status_code == 200:
                    # SUCCESS - report and return
                    await account_manager.report_success(account.id, request_data.model)

                    openai_response = await collect_stream_response(
                        http_client.client,
                        response,
                        request_data.model,
                        model_cache,
                        auth_manager,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer
                    )

                    await http_client.close()
                    logger.info(f"HTTP 200 - POST /v1/chat/completions (non-streaming) - completed")

                    if debug_logger:
                        debug_logger.discard_buffers()

                    return JSONResponse(content=openai_response)
                
                else:
                    # ERROR - classify and decide
                    try:
                        error_content = await response.aread()
                    except Exception:
                        error_content = b"Unknown error"
                    
                    await http_client.close()
                    error_text = error_content.decode('utf-8', errors='replace')
                    
                    # Extract error reason and save for final return
                    error_reason = None
                    try:
                        error_json = json.loads(error_text)
                        from kiro.kiro_errors import enhance_kiro_error
                        error_info = enhance_kiro_error(error_json)
                        error_reason = error_info.reason
                        last_error_message = error_info.user_message
                        last_error_status = response.status_code
                        logger.debug(f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})")
                    except (json.JSONDecodeError, KeyError):
                        last_error_message = error_text
                        last_error_status = response.status_code
                    
                    # Classify error
                    error_type = classify_error(response.status_code, error_reason)
                    
                    if error_type == ErrorType.FATAL:
                        # FATAL - return to client immediately
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type,
                            response.status_code, error_reason
                        )
                        
                        logger.warning(f"HTTP {response.status_code} - POST /v1/chat/completions - {last_error_message[:100]}")
                        
                        if debug_logger:
                            debug_logger.flush_on_error(response.status_code, last_error_message)
                        
                        return JSONResponse(
                            status_code=response.status_code,
                            content={
                                "error": {
                                    "message": last_error_message,
                                    "type": "kiro_api_error",
                                    "code": response.status_code
                                }
                            }
                        )
                    
                    else:  # ErrorType.RECOVERABLE
                        # RECOVERABLE - try next account
                        await account_manager.report_failure(
                            account.id, request_data.model, error_type,
                            response.status_code, error_reason
                        )
                        
                        # Single account - no point in failover, break immediately
                        if len(all_accounts) == 1:
                            break
                        
                        continue  # Next iteration
            
            except HTTPException as e:
                await http_client.close()
                
                # Network errors (502/504 from request_with_retry) = RECOVERABLE
                # These are thrown ONLY for network-level issues (timeouts, connection errors)
                # NOT for HTTP-level errors (which are returned as response objects)
                if e.status_code in (502, 504):
                    # Network error → try next account
                    await account_manager.report_failure(
                        account.id, request_data.model, ErrorType.RECOVERABLE,
                        e.status_code, None
                    )
                    
                    last_error_message = str(e.detail)
                    last_error_status = e.status_code
                    
                    # Single account - no point in failover, break immediately
                    if len(all_accounts) == 1:
                        break
                    
                    logger.warning(f"Network error on account {account.id}, trying next account")
                    continue  # Try next account
                
                # All other HTTPException (400, 500, etc.) = application errors
                # These come from build_kiro_payload() or other places → re-raise immediately
                logger.error(f"HTTP {e.status_code} - POST /v1/chat/completions - {e.detail}")
                if debug_logger:
                    debug_logger.flush_on_error(e.status_code, str(e.detail))
                raise
            except Exception as e:
                await http_client.close()
                logger.error(f"Internal error: {e}", exc_info=True)
                logger.error(f"HTTP 500 - POST /v1/chat/completions - {str(e)[:100]}")
                if debug_logger:
                    debug_logger.flush_on_error(500, str(e))
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
        # All attempts exhausted
        if len(all_accounts) == 1:
            # Single account - return its original error
            # last_error_status and last_error_message are guaranteed to be set
            raise HTTPException(
                status_code=last_error_status,
                detail=last_error_message
            )
        else:
            # Multiple accounts - generic error with context
            detail = "All accounts failed after full circle."
            if last_error_message:
                detail += f" Last error: {last_error_message}"
            raise HTTPException(status_code=503, detail=detail)
    
    else:
        # ==============================================================================
        # LEGACY MODE: Single Account (no failover)
        # ==============================================================================
        account = request.app.state.account_manager.get_first_account()
        if not account.auth_manager:
            logger.error("No initialized accounts available (legacy mode)")
            raise HTTPException(503, "No initialized accounts available")
        auth_manager = account.auth_manager
        model_cache = account.model_cache
        model_resolver = account.model_resolver
    
    # Generate conversation ID for Kiro API (random UUID, not used for tracking)
    conversation_id = generate_conversation_id()
    
    # Build payload for Kiro
    # profileArn is only needed for Kiro Desktop auth
    # AWS SSO OIDC (Builder ID) users don't need profileArn and it causes 403 if sent
    profile_arn_for_payload = ""
    if auth_manager.auth_type == AuthType.KIRO_DESKTOP and auth_manager.profile_arn:
        profile_arn_for_payload = auth_manager.profile_arn
    
    try:
        kiro_payload = build_kiro_payload(
            request_data,
            conversation_id,
            profile_arn_for_payload
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Log Kiro payload
    try:
        kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode('utf-8')
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")
    
    # Create HTTP client with retry logic
    # For streaming: use per-request client to avoid CLOSE_WAIT leak on VPN disconnect (issue #54)
    # For non-streaming: use shared client for connection pooling
    url = f"{auth_manager.api_host}/generateAssistantResponse"
    logger.debug(f"Kiro API URL: {url}")
    
    if request_data.stream:
        # Streaming mode: per-request client prevents orphaned connections
        # when network interface changes (VPN disconnect/reconnect)
        http_client = KiroHttpClient(auth_manager, shared_client=None)
    else:
        # Non-streaming mode: shared client for efficient connection reuse
        shared_client = request.app.state.http_client
        http_client = KiroHttpClient(auth_manager, shared_client=shared_client)
    try:
        # Prepare data for fallback token counting (needed by both paths)
        messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
        tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None

        if request_data.stream:
            # Streaming: do NOT pre-await the upstream request.
            # Pre-awaiting blocks byte-silent for the entire prefill (TTFT) window
            # before StreamingResponse is returned, so large-context follow-ups
            # (e.g. after a tool call) look like a dead stream to the client.
            # Make the request INSIDE the generator where keep-alive SSE comments
            # flow during prefill (see stream_with_first_token_retry).
            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                try:
                    # Create retry request function (used for first attempt + retries)
                    async def make_retry_request():
                        return await http_client.request_with_retry(
                            "POST", url, kiro_payload, stream=True
                        )

                    # web_search 2nd round-trip: rebuild payload with the original
                    # messages + the model's tool_call + the search results, then
                    # re-request so the model can read the results and answer.
                    # strip_web_search=True (final round) removes the web_search tool
                    # so the model MUST answer instead of looping on more searches.
                    async def make_followup_request(followup_messages, strip_web_search=False):
                        from kiro.models_openai import ChatMessage
                        update = {
                            "messages": [ChatMessage(**m) for m in followup_messages],
                            "stream": True,
                        }
                        if strip_web_search and request_data.tools:
                            kept_tools = [
                                t for t in request_data.tools
                                if ((t.function.name if t.function else None) or t.name)
                                != "web_search"
                            ]
                            update["tools"] = kept_tools or None
                            if not kept_tools:
                                update["tool_choice"] = None
                        fu_request = request_data.model_copy(update=update)
                        fu_conversation_id = generate_conversation_id()
                        fu_payload = build_kiro_payload(
                            fu_request, fu_conversation_id, profile_arn_for_payload
                        )
                        return await http_client.request_with_retry(
                            "POST", url, fu_payload, stream=True
                        )

                    # initial_response=None forces the request to happen inside the
                    # retry wrapper, which sends keepalive chunks during prefill.
                    async for chunk in stream_with_first_token_retry(
                        make_request=make_retry_request,
                        client=http_client.client,
                        model=request_data.model,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                        initial_response=None,
                        request_messages=messages_for_tokenizer,
                        request_tools=tools_for_tokenizer,
                        make_followup_request=make_followup_request
                    ):
                        yield chunk
                except GeneratorExit:
                    # Client disconnected - this is normal
                    client_disconnected = True
                    logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
                except Exception as e:
                    streaming_error = e
                    # The downstream StreamingResponse has already committed HTTP 200
                    # (status + headers were sent the moment we started yielding, and
                    # keepalive comments may have flowed during prefill). We therefore
                    # CANNOT surface this as an HTTP error status anymore. Re-raising
                    # here makes Starlette throw "response already started". Instead,
                    # emit the error in-band as an OpenAI SSE chunk + [DONE] and end
                    # the generator cleanly. The finally block still logs it as 500.
                    try:
                        import time as _t
                        err_chunk = {
                            "id": "error",
                            "object": "chat.completion.chunk",
                            "created": int(_t.time()),
                            "model": request_data.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n[kiro-gateway error] {e}"},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception:
                        pass  # Client already disconnected
                    # Do NOT re-raise: response already started.
                    return
                finally:
                    await http_client.close()
                    # Log access log for streaming (success or error)
                    if streaming_error:
                        error_type = type(streaming_error).__name__
                        error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                        logger.error(f"HTTP 500 - POST /v1/chat/completions (streaming) - [{error_type}] {error_msg[:100]}")
                    elif client_disconnected:
                        logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - client disconnected")
                    else:
                        logger.info(f"HTTP 200 - POST /v1/chat/completions (streaming) - completed")
                    # Write debug logs AFTER streaming completes
                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(500, str(streaming_error))
                        else:
                            debug_logger.discard_buffers()

            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

        # Non-streaming: pre-await so we can classify upstream errors before
        # committing to a response shape.
        # Important: we wait for Kiro response BEFORE returning,
        # so that 200 OK means Kiro accepted the request and started responding
        response = await http_client.request_with_retry(
            "POST",
            url,
            kiro_payload,
            stream=True
        )

        if response.status_code != 200:
            try:
                error_content = await response.aread()
            except Exception:
                error_content = b"Unknown error"
            
            await http_client.close()
            error_text = error_content.decode('utf-8', errors='replace')
            
            # Try to parse JSON response from Kiro to extract error message
            error_message = error_text
            try:
                error_json = json.loads(error_text)
                # Enhance Kiro API errors with user-friendly messages
                from kiro.kiro_errors import enhance_kiro_error
                error_info = enhance_kiro_error(error_json)
                error_message = error_info.user_message
                # Log original error for debugging
                logger.debug(f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})")
            except (json.JSONDecodeError, KeyError):
                pass
            
            # Log access log for error (before flush, so it gets into app_logs)
            logger.warning(
                f"HTTP {response.status_code} - POST /v1/chat/completions - {error_message[:100]}"
            )
            
            # Flush debug logs on error ("errors" mode)
            if debug_logger:
                debug_logger.flush_on_error(response.status_code, error_message)
            
            # Return error in OpenAI API format
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": {
                        "message": error_message,
                        "type": "kiro_api_error",
                        "code": response.status_code
                    }
                }
            )

        # Non-streaming mode - collect entire response
        openai_response = await collect_stream_response(
            http_client.client,
            response,
            request_data.model,
            model_cache,
            auth_manager,
            request_messages=messages_for_tokenizer,
            request_tools=tools_for_tokenizer
        )
        
        await http_client.close()
        
        # Log access log for non-streaming success
        logger.info(f"HTTP 200 - POST /v1/chat/completions (non-streaming) - completed")

        # Write debug logs after non-streaming request completes
        if debug_logger:
            debug_logger.discard_buffers()

        return JSONResponse(content=openai_response)
    
    except HTTPException as e:
        await http_client.close()
        
        # Network errors (502/504 from request_with_retry) = RECOVERABLE
        # In legacy mode, we still log them but re-raise (no failover available)
        if e.status_code in (502, 504):
            logger.warning(f"Network error (legacy mode, no failover available)")
        
        # Log access log for HTTP error
        logger.error(f"HTTP {e.status_code} - POST /v1/chat/completions - {e.detail}")
        # Flush debug logs on HTTP error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error: {e}", exc_info=True)
        # Log access log for internal error
        logger.error(f"HTTP 500 - POST /v1/chat/completions - {str(e)[:100]}")
        # Flush debug logs on internal error ("errors" mode)
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")