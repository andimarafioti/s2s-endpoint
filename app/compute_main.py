from app.compute_app import ComputeSettings, create_app

settings = ComputeSettings.from_env()
app = create_app(settings)
