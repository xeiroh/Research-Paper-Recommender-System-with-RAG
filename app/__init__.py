# app/__init__.py
from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st

@st.cache_resource(show_spinner="🔍 Loading FAISS Index…")
def get_faiss_index(path=".cache/faiss_index.index"):
	import faiss
	return faiss.read_index(path)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(ROOT, ".env"))

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
	# --- OpenAI ---
	openai_api_key: str = Field(..., env="OPENAI_API_KEY")
	openai_chat_model: str = Field("gpt-5", env="OPENAI_CHAT_MODEL")
	openai_embed_model: str = Field("text-embedding-3-small", env="OPENAI_EMBED_MODEL")


	# --- Paths / IO ---
	root: str = ROOT
	data_dir: str = Field("data", env="DATA_DIR")
	cache_dir: str = Field(".cache", env="CACHE_DIR")
	app_env: str = Field("dev", env="APP_ENV")
	log_level: str = Field("INFO", env="LOG_LEVEL")
	seed: int = Field(42, env="SEED")

	# --- App ---
	port: int = Field(8501, env="PORT")

	@field_validator("openai_api_key")
	@classmethod
	def _must_be_real_key(cls, v: str) -> str:
		if not v or v == "..." or v.strip() == "":
			raise ValueError("OPENAI_API_KEY not loaded from .env")
		return v


	class Config:
		case_sensitive = False
		env_file = ".env"
		env_file_encoding = "utf-8"
		extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
	s = Settings()  # reads .env automatically
	logging.basicConfig(level=getattr(logging, s.log_level.upper(), logging.INFO))
	return s


# Handy single import
settings = get_settings()
# print(os.getenv("OPENAI_API_KEY"))

__all__ = ["Settings", "get_settings", "settings"]
__version__ = "0.1.0"
