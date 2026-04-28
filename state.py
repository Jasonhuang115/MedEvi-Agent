from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PICOQuery(BaseModel):
    """PICOS查询结构"""
    population: str = Field(default="", description="目标人群")
    intervention: str = Field(default="", description="干预措施")
    comparison: str = Field(default="", description="对照措施")
    outcome: str = Field(default="", description="结局指标")
    study_type: str = Field(default="", description="研究类型")


class Paper(BaseModel):
    """单篇文献结构"""
    pmid: str = Field(description="PubMed ID")
    title: str = Field(default="", description="文献标题")
    abstract: str = Field(default="", description="摘要内容")
    decision: str = Field(default="", description="筛选决策: Include/Exclude")
    reason: str = Field(default="", description="纳入/排除理由")
    confidence: float = Field(default=0.0, description="置信度 0-1")


class ExtractedPICOS(BaseModel):
    """提取的PICOS实体"""
    pmid: str = Field(description="PubMed ID")
    population: str = Field(default="", description="提取的人群")
    intervention: str = Field(default="", description="提取的干预措施")
    comparison: str = Field(default="", description="提取的对照措施")
    outcome: str = Field(default="", description="提取的结局指标")
    study_type: str = Field(default="", description="提取的研究类型")
    extraction_source: str = Field(default="local", description="提取来源: local/claude")


class QuantitativeOutcome(BaseModel):
    """单篇文献的数值结局数据（LLM从摘要提取）"""
    pmid: str = Field(description="PubMed ID")
    outcome_label: str = Field(default="", description="结局指标描述")
    outcome_type: str = Field(default="binary", description="binary / continuous / time-to-event")
    # 二分类结局
    treatment_n: Optional[int] = Field(default=None, description="治疗/暴露组总人数")
    treatment_events: Optional[int] = Field(default=None, description="治疗/暴露组事件数")
    control_n: Optional[int] = Field(default=None, description="对照组总人数")
    control_events: Optional[int] = Field(default=None, description="对照组事件数")
    # 连续型结局
    treatment_mean: Optional[float] = Field(default=None)
    treatment_sd: Optional[float] = Field(default=None)
    control_mean: Optional[float] = Field(default=None)
    control_sd: Optional[float] = Field(default=None)
    # 效应量
    effect_measure: str = Field(default="OR", description="OR / RR / HR / MD / SMD")
    effect_size: Optional[float] = Field(default=None, description="点估计值")
    ci_lower: Optional[float] = Field(default=None)
    ci_upper: Optional[float] = Field(default=None)
    p_value: Optional[float] = Field(default=None)
    # 遗传模型（仅遗传关联研究）
    genetic_model: str = Field(default="", description="allelic / dominant / recessive / additive")
    # 质量标记
    extraction_confidence: str = Field(default="MEDIUM", description="HIGH / MEDIUM / LOW")
    needs_review: bool = Field(default=False, description="是否需要人工复核")


class PaperState(BaseModel):
    """LangGraph共享状态 - 全链路数据流转格式"""
    query: str = Field(default="", description="用户的检索问题")
    pico_query: Optional[PICOQuery] = Field(default=None, description="解析后的PICOS查询词")
    pubmed_ids: List[str] = Field(default_factory=list, description="PubMed检索返回的PMID列表")
    raw_abstracts: List[Dict] = Field(default_factory=list, description="原始摘要列表")
    reranked_abstracts: List[Dict] = Field(default_factory=list, description="重排序后的摘要列表")
    screened_papers: List[Paper] = Field(default_factory=list, description="筛选后纳入的文献")
    extracted_picos: List[ExtractedPICOS] = Field(default_factory=list, description="提取的PICOS实体")
    quantitative_outcomes: List[QuantitativeOutcome] = Field(default_factory=list, description="提取的数值结局数据")
    grade_report: str = Field(default="", description="GRADE证据分级报告")
    error: str = Field(default="", description="错误信息")
