from django.apps import AppConfig
from sentence_transformers import SentenceTransformer

class CsasConfig(AppConfig):
    name = 'csas'
    model = None

    def ready(self):
        if not self.model:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Open-source model loaded successfully.")
