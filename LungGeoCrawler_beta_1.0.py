import datetime
import time
import pandas as pd
from Bio import Entrez
from tqdm import tqdm
import json
import re
import requests
from bs4 import BeautifulSoup
import os
import sys
from dotenv import load_dotenv
from http import HTTPStatus
from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError


# 使用脚本所在目录作为工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)


# =================配置区域=================
# 1. NCBI登陆邮箱
Entrez.email = "YOUR_EMAIL"

# 2. API Key，填在这里或环境变量中
load_dotenv(override=True)

# 3. ModelScope OpenAI 兼容接口，请填写 MODELSCOPE_ACCESS_TOKEN
MODELSCOPE_ACCESS_TOKEN = os.getenv("MODELSCOPE_ACCESS_TOKEN")

client = OpenAI(
    api_key=MODELSCOPE_ACCESS_TOKEN,  # ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/" # ModelScope OpenAI 兼容接口地址
)

# =================功能区域=================

class GeoMetadataCrawler:
    def __init__(self, use_ai: bool = False):
        self.use_ai = use_ai

    def search_geo(self, query, max_results=200):
        """
        根据关键词搜索 GEO，在搜索阶段即跳过白名单，返回指定数量的未处理 ID。
        """
        print(f"正在搜索NCBI: {query} ...")
        whitelist = self.load_whitelist()
        if whitelist:
            print(f"白名单共 {len(whitelist)} 条，将直接跳过已处理 GSE。")

        filtered_ids = []
        retstart = 0
        step = 200  # 批量抓取，减少往返

        try:
            while len(filtered_ids) < max_results:
                handle = Entrez.esearch(db="gds", term=query, retmax=step, retstart=retstart)
                results = Entrez.read(handle)
                handle.close()

                id_list = results.get("IdList", [])
                total_count = int(results.get("Count", 0))
                if not id_list:
                    break

                try:
                    summary_handle = Entrez.esummary(db="gds", id=",".join(id_list))
                    summaries = Entrez.read(summary_handle)
                    summary_handle.close()
                except Exception as e:
                    print(f"⚠️ esummary 获取失败（retstart={retstart}）: {e}")
                    break

                for doc in summaries:
                    accession = doc.get("Accession", "")
                    doc_id = doc.get("Id")
                    if not accession.startswith("GSE"):
                        continue
                    if accession in whitelist:
                        continue
                    if doc_id:
                        filtered_ids.append(doc_id)
                        if len(filtered_ids) >= max_results:
                            break

                retstart += len(id_list)
                if retstart >= total_count:
                    break

            print(f" 找到 {len(filtered_ids)} 个未在白名单中的数据集（请求上限 {max_results}）。")
            return filtered_ids
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []

    def load_whitelist(self, whitelist_file="whitelist.csv"):
        """
        读取白名单列表，用于跳过已处理的 GSE
        """
        whitelist_path = os.path.join(SCRIPT_DIR, whitelist_file)
        try:
            if not os.path.exists(whitelist_path):
                print(f" 未找到白名单文件 {whitelist_file}，将处理全部结果")
                return set()
            with open(whitelist_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            print(f"⚠️ 白名单读取失败: {e}")
            return set()

    def fetch_pubmed_metadata(self, pubmed_ids):
        """
        使用 Bio.Entrez.efetch 从 PubMed 获取文献元数据
        返回: (citation_str, doi, journal_name, study_text_context)
        """
        if not pubmed_ids:
            return "", "", "", ""
        
        # 如果是单个 ID，转换为列表
        if isinstance(pubmed_ids, str):
            pubmed_ids = [pubmed_ids]
        
        from xml.etree import ElementTree as ET
        
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(pubmed_ids),
                rettype="xml",
                retmode="xml"
            )
            xml_content = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_content)
            citations = []
            doi = ""
            journal_name = ""
            abstract_text = ""
            pmc_id = ""
            
            for article in root.findall(".//PubmedArticle"):
                # 提取作者
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        author_name = lastname.text
                        if forename is not None:
                            author_name = f"{lastname.text} {forename.text[0]}"
                        authors.append(author_name)
                
                author_str = authors[0] + " et al." if len(authors) > 1 else (authors[0] if authors else "Unknown")
                
                # 提取标题
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # 提取期刊名称
                journal_elem = article.find(".//Journal/Title")
                if journal_elem is not None and journal_elem.text:
                    journal_name = journal_elem.text
                
                # 提取年份
                year_elem = article.find(".//PubDate/Year")
                if year_elem is None:
                    year_elem = article.find(".//PubDate/MedlineDate")
                year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""
                
                # 提取 DOI 和 PMCID
                for id_elem in article.findall(".//ArticleId"):
                    id_type = id_elem.get("IdType")
                    if id_type == "doi" and not doi:
                        doi = id_elem.text
                    elif id_type == "pmc" and not pmc_id:
                        pmc_id = id_elem.text
                
                # 提取摘要
                abstract_parts = []
                for abstract_text_elem in article.findall(".//AbstractText"):
                    label = abstract_text_elem.get("Label", "")
                    text = abstract_text_elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract_text = " ".join(abstract_parts)
                
                # 提取 PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # 构建引用字符串
                citation_parts = []
                if author_str:
                    citation_parts.append(author_str)
                if title:
                    citation_parts.append(title)
                if journal_name:
                    citation_parts.append(journal_name)
                if year:
                    citation_parts.append(f"({year})")
                if pmid:
                    citation_parts.append(f"PMID: {pmid}")
                
                citations.append(" ".join(citation_parts))
            
            # 尝试从 PMC 获取全文
            study_text_context = ""
            if pmc_id:
                study_text_context = self._fetch_pmc_fulltext(pmc_id)
            
            # 如果没有 PMC 全文，使用摘要
            if not study_text_context and abstract_text:
                study_text_context = f"[Abstract] {abstract_text}"
            
            return "; ".join(citations), doi, journal_name, study_text_context
            
        except Exception as e:
            print(f"⚠️ PubMed 元数据获取失败: {e}")
            return f"PubMed: {', '.join(pubmed_ids)}", "", "", ""
    
    def _fetch_pmc_fulltext(self, pmc_id):
        """
        从 PubMed Central 获取全文（聚焦 Methods/Results）
        限制返回 8000 字符
        """
        from xml.etree import ElementTree as ET
        
        try:
            # 清理 PMC ID 格式
            pmc_id_clean = pmc_id.replace("PMC", "")
            
            handle = Entrez.efetch(
                db="pmc",
                id=pmc_id_clean,
                rettype="xml",
                retmode="xml"
            )
            xml_content = handle.read()
            handle.close()
            
            root = ET.fromstring(xml_content)
            
            # 提取感兴趣的章节 (Methods, Results, Materials)
            sections_of_interest = ["methods", "materials", "results", "patients", "cohort", "samples"]
            extracted_text = []
            
            for sec in root.findall(".//sec"):
                title_elem = sec.find("title")
                if title_elem is not None and title_elem.text:
                    title_lower = title_elem.text.lower()
                    if any(keyword in title_lower for keyword in sections_of_interest):
                        # 提取该章节的所有文本
                        section_text = []
                        for p in sec.findall(".//p"):
                            if p.text:
                                section_text.append(p.text)
                            # 也获取子元素的文本
                            for child in p:
                                if child.text:
                                    section_text.append(child.text)
                                if child.tail:
                                    section_text.append(child.tail)
                        
                        if section_text:
                            extracted_text.append(f"[{title_elem.text}] " + " ".join(section_text))
            
            # 如果没有找到特定章节，尝试获取 body 中的所有文本
            if not extracted_text:
                body = root.find(".//body")
                if body is not None:
                    all_text = []
                    for p in body.findall(".//p"):
                        text = "".join(p.itertext())
                        if text:
                            all_text.append(text)
                    extracted_text = [" ".join(all_text)]
            
            full_text = " ".join(extracted_text)
            
            # 限制为 8000 字符
            if len(full_text) > 8000:
                full_text = full_text[:8000] + "..."
            
            return f"[PMC Full Text] {full_text}" if full_text else ""
            
        except Exception as e:
            print(f"⚠️ PMC 全文获取失败 ({pmc_id}): {e}")
            return ""

    def fetch_gse_page_text(self, gse_id):
        """
        直接爬取 GSE 网页的 HTML，并抽取静态元数据
        """
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        
        # 重试机制
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code != 200:
                    print(f"⚠️ 网页请求返回状态码 {response.status_code} (GSE: {gse_id})")
                    return None

                soup = BeautifulSoup(response.text, "html.parser")

                def get_text_by_label(label_text):
                    td = soup.find("td", string=re.compile(label_text, re.IGNORECASE))
                    if td:
                        next_td = td.find_next_sibling("td")
                        return next_td.get_text(strip=True) if next_td else ""
                    return ""
                
                summary_text = get_text_by_label("Summary")
                design_text = get_text_by_label("Overall design")
                submission_date = get_text_by_label("Submission date")

                # 提取 Organization / Contact name
                organization = get_text_by_label("Organization name")
                if not organization:
                    organization = get_text_by_label("Contact name")


                # SRA 状态
                page_text_all = soup.get_text(" ", strip=True)
                sra_status = "Yes" if ("SRA Run Selector" in page_text_all or "Traces/sra" in str(soup)) else "No"

                # 3. Citation 和 DOI 提取 - 使用 PubMed API 获取完整信息
                citation_text = ""
                pubmed_ids = []
                fetched_doi = ""
                
                # 查找包含 "Citation" 的 td 标签
                citation_td = soup.find("td", string=re.compile(r"Citation", re.IGNORECASE))
                if citation_td:
                    parent_tr = citation_td.find_parent("tr")
                    if parent_tr:
                        next_td = citation_td.find_next_sibling("td")
                        if next_td:
                            # 提取 PubMed ID (从链接或纯数字文本)
                            for link in next_td.find_all("a", href=True):
                                href = link.get("href", "")
                                if "pubmed" in href:
                                    pmid = re.search(r"pubmed/(\d+)", href)
                                    if pmid:
                                        pubmed_ids.append(pmid.group(1))
                            # 检查纯数字文本
                            text_content = next_td.get_text(" ", strip=True)
                            if re.match(r"^\d+$", text_content.strip()):
                                if text_content.strip() not in pubmed_ids:
                                    pubmed_ids.append(text_content.strip())
                
                # 使用 PubMed API 获取完整引用、DOI、期刊名和研究文本
                journal_name = ""
                study_text_context = ""
                if pubmed_ids:
                    citation_text, fetched_doi, journal_name, study_text_context = self.fetch_pubmed_metadata(pubmed_ids)
                
                if not citation_text and pubmed_ids:
                    citation_text = f"PubMed: {', '.join(pubmed_ids)}"

                # 5. 提取样本列表 (提取前100个样本名给AI，帮助判断是否配对)
                samples_text = []
                sample_rows = soup.find_all("tr")
                for row in sample_rows:
                    links = row.find_all("a", href=True)
                    for link in links:
                        if "acc=GSM" in link["href"]:
                            samples_text.append(f"{link.get_text(strip=True)} | {row.get_text(strip=True)}")

                samples_preview = "; ".join(samples_text[:100])

                # 提取 This SuperSeries is composed of the following SubSeries
                subseries_ids = []
                subseries_contents = []  # 存储每个子数据集的完整描述
                has_subseries = False
                
                # 查找包含 "SubSeries" 关键词的 <td> 标签
                subseries_td = soup.find("td", string=re.compile(r"SubSeries", re.IGNORECASE))
                if subseries_td:
                    has_subseries = True
                    # 在同一 <tr> 或后续兄弟元素中查找 GSE 链接
                    parent_tr = subseries_td.find_parent("tr")
                    if parent_tr:
                        # 遍历后续所有行，直到遇到新的板块
                        current_row = parent_tr.find_next_sibling("tr")
                        while current_row:
                            # 查找所有 GSE 开头的链接
                            links = current_row.find_all("a", href=True)
                            gse_found_in_row = False
                            for link in links:
                                if "acc=GSE" in link.get("href", ""):
                                    gse_match = re.search(r"(GSE\d+)", link.get("href", ""))
                                    if gse_match:
                                        gse_id_sub = gse_match.group(1)
                                        subseries_ids.append(gse_id_sub)
                                        # 提取该行的完整文本内容
                                        row_text = current_row.get_text(" ", strip=True)
                                        subseries_contents.append(f"{gse_id_sub}: {row_text}")
                                        gse_found_in_row = True
                            # 如果这一行没有 GSE 链接，检查是否是新板块的开始
                            if not gse_found_in_row:
                                # 检查是否有新的标签行（通常包含粗体或特定格式）
                                tds = current_row.find_all("td")
                                if tds and len(tds) > 0:
                                    first_td_text = tds[0].get_text(strip=True)
                                    # 如果第一个 td 看起来像是新的标签，停止
                                    if first_td_text and not first_td_text.startswith("GSE"):
                                        break
                            current_row = current_row.find_next_sibling("tr")
                
                # 另一种查找方式：直接搜索页面中所有 SubSeries 相关的 GSE 链接
                if not subseries_ids:
                    # 查找包含 "SubSeries" 文字的区域
                    for text_node in soup.find_all(string=re.compile(r"SubSeries", re.IGNORECASE)):
                        has_subseries = True
                        parent = text_node.find_parent("table")
                        if parent:
                            for link in parent.find_all("a", href=True):
                                if "acc=GSE" in link.get("href", ""):
                                    gse_match = re.search(r"(GSE\d+)", link.get("href", ""))
                                    if gse_match and gse_match.group(1) not in subseries_ids:
                                        gse_id_sub = gse_match.group(1)
                                        subseries_ids.append(gse_id_sub)
                                        # 提取父行的文本内容
                                        parent_row = link.find_parent("tr")
                                        if parent_row:
                                            row_text = parent_row.get_text(" ", strip=True)
                                            subseries_contents.append(f"{gse_id_sub}: {row_text}")
                
                subseries_str = "; ".join(subseries_ids) if subseries_ids else ""
                subseries_content_str = " | ".join(subseries_contents) if subseries_contents else ""

                return {
                    "summary": summary_text,
                    "overall_design": design_text,
                    "samples_preview": samples_preview,
                    "submission_date": submission_date,
                    "organization": organization,
                    "sra_status": sra_status,
                    "citation": citation_text,
                    "pubmed_doi": fetched_doi,  # 从 PubMed 获取的 DOI
                    "pubmed_ids": pubmed_ids,  # PubMed ID 列表
                    "journal_name": journal_name,  # 期刊名称
                    "study_text_context": study_text_context,  # PMC 全文或摘要
                    "has_subseries": "Yes" if has_subseries else "No",
                    "subseries_ids": subseries_str,
                    "subseries_content": subseries_content_str
                }
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # 失败后等待2秒再试
                    continue
                else:
                    print(f"⚠️ 网页爬取失败 {gse_id} (已重试{max_retries}次): {e}")
                    return None
            except Exception as e:
                print(f"⚠️ 解析异常 {gse_id}: {e}")
                return None

    def ai_extract_info(self, title, summary, design, citation_text="", samples="", subseries_content="", journal_name="", study_text_context=""):
        """使用 OpenAI 兼容接口提取结构化信息，包含期刊影响因子和研究架构分析"""
        if not self.use_ai:
            return {}
        if not MODELSCOPE_ACCESS_TOKEN:
            print("⚠️ 未设置 MODELSCOPE_ACCESS_TOKEN，跳过 AI 解析")
            return {}

        # 保护性截断，减少Token消耗
        limit = 2000
        subseries_limit = 1000
        study_text_limit = 5000  # PMC 全文限制

        # 封装给模型的上下文
        text_content = (
            f"Title: {title}\n"
            f"Summary: {summary[:limit]}\n"
            f"Design: {design[:limit]}\n"
            f"Citation_String: {citation_text}\n"
            f"Journal: {journal_name}\n"
            f"Samples: {samples[:limit]}\n"
            f"SubSeries_Content: {subseries_content[:subseries_limit]}\n"
            f"Study_Text: {study_text_context[:study_text_limit]}"
        )

        prompt_content = f"""
        You are an expert biocurator for Lung Cancer Single-cell Research.
        Analyze the study metadata and return ONLY valid JSON.

        Text to analyze:
        {text_content}

        === Extraction Rules ===
        1. **Basic Info**: Patients (int), Samples (int).
        2. **Subtypes**: Count LUAD, LUSC, SCLC (int).
        3. **Journal_Info**: 
           - "Name": Journal name.
           - "Estimated_IF": Estimate Impact Factor (float, e.g., 15.3 for Nat Commun).
        4. **Metadata**:
           - "Stage": Clinical stage (I-IV or mixed).
           - "Treatment": e.g., "Anti-PD1", "TKI", "Chemo", "Treatment Naive".
           - "Sequencing_Strategy": ["scRNA-seq", "snRNA-seq", "scATAC-seq", "Spatial", etc.].
        5. **Study_Architecture**:
           - "Sample_Topology": ["Primary Tumor", "Adjacent Normal", "Lymph Node Metastasis", "Brain Metastasis", "PBMC", "Pleural Effusion"].
           - "Design_Class": One of ["Cross-sectional (Tumor vs Normal)", "Longitudinal (Pre/Post Treatment)", "Multi-region (Intra-tumor heterogeneity)", "Pan-cancer Atlas", "Other"].
           - "Treatment_Context": e.g., "Neoadjuvant Anti-PD1", "TKI Resistance", "Treatment Naive".
           - "Dataset_Composition": Brief cohort summary.
        6. **Experiment**: Has_Paired_Normal (bool), Tissue_Source.

        === JSON Output Format ===
        {{{{
            "Patients": int,
            "LUAD": int,
            "LUSC": int,
            "SCLC": int,
            "Other_Subtype_Info": "string",
            "Samples": int,
            "Composition": "string",
            "Stage": "string",
            "Treatment": "string",
            "Sorting": "string",
            "Platform": "string",
            "DOI": "string",
            "Sequencing_Strategy": ["string"],
            "Tissue_Source": "string",
            "Journal_Info": {{{{"Name": "string", "Estimated_IF": float}}}},
            "Study_Architecture": {{{{
                "Sample_Topology": ["string"],
                "Design_Class": "string",
                "Treatment_Context": "string",
                "Dataset_Composition": "string"
            }}}},
            "Experimental_Design": {{{{
                "Has_Paired_Normal": bool,
                "Has_Blood_Pair": bool,
                "Has_Treatment_Pair": bool,
                "Design_Focus": "string"
            }}}}
        }}}}
            
        Return ONLY the JSON object.
        """

        #  === OpenAI 兼容接口路径  ===
        @retry(
            stop=stop_after_attempt(5),  # 5 次重试
            wait=wait_exponential(multiplier=2, min=10, max=120),  # 指数退避: 10s, 20s, 40s, 80s, 120s
            retry=retry_if_exception_type(RateLimitError),  # 只在 RateLimitError 时重试
            reraise=True
        )
        def call_model_scope(prompt_content): 
            return client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[
                    {"role": "system", "content": "You are a rigid JSON extractor for bioinformatics metadata. You strictly output valid JSON only. Do not wrap result in markdown blocks."},
                    {"role": "user", "content": prompt_content},
                    ],
                    temperature=0.1,
                )
        try:
            completion = call_model_scope(prompt_content)
            # 请求成功后添加较长延迟，给服务器更多冷却时间
            time.sleep(10)
            content = completion.choices[0].message.content
            # 清理markdown标记
            content = re.sub(r'^```json\s*', "", content)
            content = re.sub(r'\s*```$', "", content)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        return {}
                return {}
        except RetryError as e:
            # 捕获 Tenacity 放弃重试后的异常
            original_error = e.last_attempt.exception() if e.last_attempt else e
            print(f"❌ AI 请求失败 (已耗尽重试次数): {title} - 错误原因: {original_error}")
            time.sleep(20)  # 额外等待
            return {}
        except RateLimitError:
            # 双重保险，防止未被 wrap 的原始异常漏网
            print(f"⚠️ AI 请求频率过高 (Rate Limit): {title}")
            time.sleep(20)
            return {}
        except Exception as e:
            error_msg = str(e)
            if "BadRequest" in error_msg or "400" in error_msg:
                print(f"⚠️ AI 请求参数错误 (可能是文本过长): {title}")
            else:
                print(f"⚠️ AI 解析未处理异常: {e}")
            return {}

    # ================数据处理区域=================
    def process_data(self, id_list):
        """整合数据并对齐 Excel 格式，带断点续传机制"""
        data = []
        print("AI正在解析 ...")
        
        # 临时文件用于断点续传
        temp_file = f"temp_crawl_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        checkpoint_interval = 4  # 每处理 4 个样本保存一次

        try:
            handle = Entrez.esummary(db="gds", id=",".join(id_list))
            summaries = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"❌ 获取 Accession 失败: {e}")
            return pd.DataFrame()

        whitelist = self.load_whitelist()
        if whitelist:
            print(f" 白名单共 {len(whitelist)} 条，将跳过已处理 GSE。")

        processed_count = 0
        for doc in tqdm(summaries, desc="Processing GSEs"):
            accession = doc.get("Accession", "")
            gse_id = accession
            if not gse_id.startswith("GSE") or gse_id in whitelist:
                continue

            page_data = self.fetch_gse_page_text(gse_id)
            if not page_data:
                print(f"⚠️ 无法获取 {gse_id} 的网页内容")
                continue

            ai_data = {}
            if self.use_ai:
                ai_data = self.ai_extract_info(
                    gse_id, 
                    page_data["summary"], 
                    page_data["overall_design"], 
                    page_data["citation"], 
                    page_data["samples_preview"],
                    page_data.get("subseries_content", ""),
                    page_data.get("journal_name", ""),
                    page_data.get("study_text_context", "")
                )

            # 3. 数据映射
            exp_design = ai_data.get("Experimental_Design", {})
            journal_info = ai_data.get("Journal_Info", {})
            study_arch = ai_data.get("Study_Architecture", {})
            seq_list = ai_data.get("Sequencing_Strategy", [])
            seq_str = "; ".join(seq_list) if isinstance(seq_list, list) else str(seq_list)

            def _yn(val): return "Yes" if val else "No"

            row = {
                "Accession": gse_id,
                "Link": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}",
                "Submission_Date": page_data["submission_date"],
                "Database": "GEO",
                "SRA": page_data["sra_status"], 
                "Citation": page_data["citation"], 
                # DOI: 优先使用 PubMed 获取的，否则使用 AI 提取的
                "DOI": page_data.get("pubmed_doi") or ai_data.get("DOI", "N/A"),
                "Organization": page_data["organization"],
                "Patients": ai_data.get("Patients", 0),
                "LUAD": ai_data.get("LUAD", 0),
                "LUSC": ai_data.get("LUSC", 0),
                "SCLC": ai_data.get("SCLC", 0),
                "Other_Subtype_Info": ai_data.get("Other_Subtype_Info", ""),
                "Samples": ai_data.get("Samples", 0),
                "Stage": ai_data.get("Stage", ""),
                "Treatment": ai_data.get("Treatment", ""),
                "Composition": ai_data.get("Composition", "N/A"),
                "Platform": ai_data.get("Platform", ""),
                "Sequencing_Strategy": seq_str,
                "Tissue_Source": ai_data.get("Tissue_Source", ""),
                "Cell sorting": ai_data.get("Sorting", "N/A"),
                "Paired_Normal": _yn(exp_design.get("Has_Paired_Normal")),
                "Design_Blood_Pair": _yn(exp_design.get("Has_Blood_Pair")),
                "Design_Treatment_Pair": _yn(exp_design.get("Has_Treatment_Pair")),
                "Design_Focus": exp_design.get("Design_Focus", ""),
                "Journal_Name": journal_info.get("Name", page_data.get("journal_name", "")),
                "Estimated_IF": journal_info.get("Estimated_IF", 0),
                "Sample_Topology": "; ".join(study_arch.get("Sample_Topology", [])) if isinstance(study_arch.get("Sample_Topology"), list) else str(study_arch.get("Sample_Topology", "")),
                "Design_Class": study_arch.get("Design_Class", ""),
                "Treatment_Context": study_arch.get("Treatment_Context", ""),
                "Dataset_Composition": study_arch.get("Dataset_Composition", ""),
                "Has_SubSeries": page_data.get("has_subseries", "No"),
                "SubSeries_IDs": page_data.get("subseries_ids", ""),
                "Original_Citation": page_data["citation"],
                "Original_Summary": page_data["summary"][:200],
                "Original_Design": page_data["overall_design"][:200]
            }
            data.append(row)
            processed_count += 1
            
            # 断点续传：每处理 checkpoint_interval 个样本保存一次临时文件
            if processed_count % checkpoint_interval == 0:
                temp_df = pd.DataFrame(data)
                temp_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
                print(f" 已保存临时进度: {processed_count} 条记录 -> {temp_file}")
        
        # 处理完成后删除临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f" 已清理临时文件: {temp_file}")
            
        return pd.DataFrame(data)

# =================执行入口=================
if __name__ == "__main__":
    crawler = GeoMetadataCrawler(use_ai=True)

    keyword = '("Lung Neoplasms"[MeSH Terms] OR "Lung cancer"[All Fields]) AND ("single cell"[All Fields] OR "scRNA-seq"[All Fields]) AND "Homo sapiens"[Organism] AND "gse"[Entry Type]'

    ids = crawler.search_geo(keyword, max_results=10)

    if ids:
        df_result = crawler.process_data(ids)
        output_filename = f"Crawled_scRNA_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.xlsx"
        df_result.to_excel(output_filename, index=False)
        print(f"处理完成！文件已保存为: {output_filename}")
    else:
        print("未找到相关数据。")
