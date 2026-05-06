"""SQLAlchemy ORM models for the geopolitical LLM analyzer — V2.0.

Core tables: Prompt (questions), ModelConfig (LLMs being studied),
Analysis (response text + AI-assigned 10-dimension scores).
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
    """One scored response — V2.0 with 10 sub-dimensions.

    TRS (T1-T5)  +  GBS (D1-D5)
    """

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    model_config_id = Column(Integer, ForeignKey("model_configs.id"), nullable=False)

    response_text = Column(Text, nullable=False)

    # TRS sub-dimensions (T1–T5), each 1.00–5.00
    t1_score = Column(Float, nullable=False)
    t2_score = Column(Float, nullable=False)
    t3_score = Column(Float, nullable=False)
    t4_score = Column(Float, nullable=False)
    t5_score = Column(Float, nullable=False)

    # GBS sub-dimensions (D1–D5), each 1.00–5.00
    d1_score = Column(Float, nullable=False)
    d2_score = Column(Float, nullable=False)
    d3_score = Column(Float, nullable=False)
    d4_score = Column(Float, nullable=False)
    d5_score = Column(Float, nullable=False)

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
    def trs(self) -> float:
        """TRS = (T1 + T2 + T3 + T4 + T5) / 5"""
        return round(
            (self.t1_score + self.t2_score + self.t3_score + self.t4_score + self.t5_score) / 5, 2
        )

    @property
    def gbs(self) -> float:
        """GBS = (D1 + D2 + D3 + D4 + D5) / 5"""
        return round(
            (self.d1_score + self.d2_score + self.d3_score + self.d4_score + self.d5_score) / 5, 2
        )

    @property
    def composite_bias(self) -> float:
        """Alias for GBS — retained for backward compatibility in viz/stats."""
        return self.gbs

    def __repr__(self):
        return (
            f"<Analysis(id={self.id}, prompt={self.prompt_id}, model={self.model_config_id}, "
            f"TRS={self.trs}, GBS={self.gbs})>"
        )
