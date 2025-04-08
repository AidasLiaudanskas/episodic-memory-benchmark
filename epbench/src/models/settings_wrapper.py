from pydantic_settings import BaseSettings
import os
import sys

class SettingsWrapper(BaseSettings):
    # parameters from the .env variables with missing default values
    PROXY: dict = {}
    OPENAI_API_KEY : str = ''
    ANTHROPIC_API_KEY: str = ''
    REPLICATE_API_TOKEN: str = ''
    OPENROUTER_API_KEY: str = ''
    GOOGLE_API_KEY: str = ''
    DEEPSEEK_API_KEY: str = ''
    REPO_PATH: str = ''
    env: dict = {}

    class Config:
        env_file = '.env' # default location, can be overridden
        env_file_encoding = "utf-8"
    
    def __init__(self, _env_file=None, **kwargs):
        if _env_file:
            self.Config.env_file = _env_file
        super().__init__(**kwargs)
        
        # Load all environment variables into a dictionary
        self.env = {key: getattr(self, key) for key in dir(self) 
                   if not key.startswith('_') and not callable(getattr(self, key))
                   and key not in ('Config', 'env')}
        
        # Check if REPO_PATH is set
        if not self.REPO_PATH:
            print("REPO_PATH in .env is not set", file=sys.stderr)
            
    def __getitem__(self, key):
        if key == 'REPO_PATH' and not self.REPO_PATH:
            print("REPO_PATH in .env is not set", file=sys.stderr)
        
        try:
            return self.env[key]
        except KeyError:
            raise KeyError(f"Key {key} not found in environment variables")
