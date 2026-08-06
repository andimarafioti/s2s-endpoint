from app.load_balancer_app import LoadBalancerSettings, create_app

settings = LoadBalancerSettings.from_env()
app = create_app(settings)
