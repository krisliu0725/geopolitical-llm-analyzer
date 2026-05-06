"""SQLAlchemy ORM models for the geopolitical LLM analyzer.

Core tables: Prompt (questions), ModelConfig (LLMs being studied),
Analysis (response text + AI-assigned scores).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    short_name = Column(String(200), nullable=False, unique=True)
    full_text = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, default="general")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    analyses = relationship("Analysis", back_populates="prompt", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Prompt(id={self.id}, name='{self.short_name}', cat='{self.category}')>"


class ModelConfig(Base):
    """An LLM being *studied* (the one that produced the response text)."""

    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), nullable=False)
    model_name = Column(String(200), nullable=False)
    display_name = Column(String(200), nullable=False)
    alignment = Column(String(50), nullable=False, default="Other/Unknown")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    analyses = relationship("Analysis", back_populates="model_config", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ModelConfig(id={self.id}, name='{self.display_name}', align='{self.alignment}')>"


class Analysis(Base):
    """One scored response — the central record tying a prompt + model + response + AI scores."""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    model_config_id = Column(Integer, ForeignKey("model_configs.id"), nullable=False)

    response_text = Column(Text, nullable=False)

    # TR score: 1.00–5.00, 2 decimal places
    tr_score = Column(Float, nullable=False)

    # Bias dimension scores: 1.00–5.00 each
    d1_score = Column(Float, nullable=False)
    d2_score = Column(Float, nullable=False)
    d3_score = Column(Float, nullable=False)
    d4_score = Column(Float, nullable=False)

    # AI-extracted key phrases with annotations (JSON string)
    key_phrases = Column(Text, nullable=True)

    # Free-text notes from the researcher
    notes = Column(Text, nullable=True)

    # Which AI model was used to perform the scoring
    scorer_model = Column(String(100), nullable=True)

    analyzed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    prompt = relationship("Prompt", back_populates="analyses")
    model_config = relationship("ModelConfig", back_populates="analyses")

    @property
    def composite_bias(self) -> float:
        return round((self.d1_score + self.d2_score + self.d3_score + self.d4_score) / 4, 2)

    def __repr__(self):
        return (
            f"<Analysis(id={self.id}, prompt={self.prompt_id}, model={self.model_config_id}, "
            f"TR={self.tr_score}, D1={self.d1_score}, D2={self.d2_score}, "
            f"D3={self.d3_score}, D4={self.d4_score})>"
        )
