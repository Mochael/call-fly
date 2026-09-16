import {diagnosticsHandler} from '../deploy/vercel-gateway.js';
export default (request,response)=>diagnosticsHandler(request,response);
