from app.callback_secret_recovery import log_encrypted_callback_auth_token
from app.load_balancer_app import LoadBalancerSettings, create_app

log_encrypted_callback_auth_token()
settings = LoadBalancerSettings.from_env()
app = create_app(settings)
