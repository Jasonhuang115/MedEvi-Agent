"""
PubMed检索工具
使用NCBI E-utilities API检索医学文献
"""
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class PubMedSearcher:
    """PubMed检索器"""

    def __init__(self):
        api_key = os.getenv("NCBI_API_KEY", "")
        # 过滤无效的占位符
        self.api_key = api_key if api_key and not api_key.startswith("your_") else ""
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.tool_name = "MedEviAgent"
        self.email = "user@example.com"  # NCBI要求提供邮箱

        # 有API Key时请求频率更高：10次/秒 vs 3次/秒
        self.rate_limit = 0.11 if self.api_key else 0.35

    def search(
        self,
        query: str,
        max_results: int = 50,
        date_range: Optional[Dict] = None
    ) -> List[str]:
        """
        检索PubMed获取PMID列表

        Args:
            query: 检索词
            max_results: 最大返回数量
            date_range: 日期范围 {"min": "2020/01/01", "max": "2023/12/31"}

        Returns:
            PMID列表
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "tool": self.tool_name,
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        if date_range:
            params["mindate"] = date_range.get("min", "")
            params["maxdate"] = date_range.get("max", "")

        url = f"{self.base_url}/esearch.fcgi"

        try:
            time.sleep(self.rate_limit)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pmids = data.get("esearchresult", {}).get("idlist", [])
            print(f"检索到 {len(pmids)} 篇文献")
            return pmids

        except requests.exceptions.RequestException as e:
            print(f"PubMed检索网络错误: {e}")
            raise  # 网络/HTTP错误向上传播，让前端显示真实原因
        except Exception as e:
            print(f"PubMed检索失败: {e}")
            raise

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """
        根据PMID获取摘要

        Args:
            pmids: PMID列表

        Returns:
            文献列表 [{"pmid": str, "title": str, "abstract": str, "authors": list}]
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": self.tool_name,
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/efetch.fcgi"

        try:
            time.sleep(self.rate_limit)
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"获取摘要网络错误: {e}")
            raise

        try:
            return self._parse_xml(response.text)
        except Exception as e:
            print(f"XML解析失败: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> List[Dict]:
        """解析PubMed XML响应"""
        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    # 提取PMID
                    pmid = article.find(".//PMID").text

                    # 提取标题
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""

                    # 提取摘要
                    abstract_texts = article.findall(".//Abstract/AbstractText")
                    abstract_parts = []
                    for text in abstract_texts:
                        label = text.get("Label", "")
                        content = "".join(text.itertext())
                        if label:
                            abstract_parts.append(f"{label}: {content}")
                        else:
                            abstract_parts.append(content)
                    abstract = " ".join(abstract_parts)

                    # 提取作者
                    authors = []
                    for author in article.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None:
                            name = lastname.text
                            if forename is not None:
                                name += f" {forename.text}"
                            authors.append(name)

                    # 提取发表年份
                    year = ""
                    pub_date = article.find(".//PubDate/Year")
                    if pub_date is not None:
                        year = pub_date.text

                    # 提取期刊
                    journal = ""
                    journal_elem = article.find(".//Journal/Title")
                    if journal_elem is not None:
                        journal = journal_elem.text

                    articles.append({
                        "pmid": pmid,
                        "title": title or "",
                        "abstract": abstract or "",
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                    })

                except Exception as e:
                    print(f"解析文章失败: {e}")
                    continue

        except Exception as e:
            print(f"XML解析失败: {e}")

        print(f"成功解析 {len(articles)} 篇文献摘要")
        return articles

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        检索并获取摘要（一步完成）

        Args:
            query: 检索词
            max_results: 最大返回数量

        Returns:
            文献列表
        """
        pmids = self.search(query, max_results)
        if not pmids:
            return []
        return self.fetch_abstracts(pmids)


def search_pubmed(query: str, max_results: int = 50) -> List[str]:
    """便捷函数：检索PubMed获取PMID列表"""
    searcher = PubMedSearcher()
    return searcher.search(query, max_results)


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """便捷函数：根据PMID获取摘要"""
    searcher = PubMedSearcher()
    return searcher.fetch_abstracts(pmids)


def search_and_fetch(query: str, max_results: int = 50) -> List[Dict]:
    """便捷函数：检索并获取摘要"""
    searcher = PubMedSearcher()
    return searcher.search_and_fetch(query, max_results)
