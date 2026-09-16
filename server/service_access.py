"""Private cloud gateway authentication; local development stays loopback-only."""
import hmac
import os
from starlette.responses import JSONResponse


def trusted_gateway(headers):
    expected=os.environ.get('VOICE_SERVICE_TOKEN','')
    supplied=headers.get('x-voice-service-token','')
    return bool(expected and hmac.compare_digest(supplied,expected))


def origin_allowed(headers):
    if trusted_gateway(headers):return True
    origin,host=headers.get('origin'),headers.get('host')
    return not origin or origin in {f'http://{host}',f'https://{host}'}


class ServiceAccess:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type'] not in ('http','websocket'):
            return await self.app(scope,receive,send)
        if os.environ.get('MOSHI_BACKEND')=='torch' or os.environ.get('VOICE_REQUIRE_AUTH')=='1':
            from starlette.datastructures import Headers
            if not trusted_gateway(Headers(scope=scope)):
                if scope['type']=='websocket':await send({'type':'websocket.close','code':1008})
                else:await JSONResponse({'detail':'Private voice service'},status_code=403)(scope,receive,send)
                return
        await self.app(scope,receive,send)
