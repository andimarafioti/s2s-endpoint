from app.speech_proxy_app import SpeechProxySettings, create_app

settings = SpeechProxySettings.from_env()
app = create_app(settings)
