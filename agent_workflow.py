from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypedDict

try:
    from typing import Annotated
except ImportError:  # Python 3.8 keeps Annotated in typing_extensions.
    from typing_extensions import Annotated


try:  # LangGraph is the intended production runner.
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - keeps the demo usable before deps are installed.
    END = "__end__"
    START = "__start__"
    StateGraph = None


class AgentState(TypedDict, total=False):
    run_id: str
    upload_path: str
    upload_url: str
    upload_items: list[dict]
    upload_paths: list[str]
    upload_urls: list[str]
    lesion_type: str
    output_dir: str
    enhancement_result: dict | None
    lesion_result: dict | None
    result_image_path: str | None
    result_image_url: str | None
    classification: dict | None
    vision_result: dict | None
    lesion_analysis: dict | None
    knowledge_context: list[dict] | None
    report: str | None
    report_json: dict | None
    reflection: dict | None
    report_revision_count: int
    report_revision_max: int
    processed_images: list[dict] | None
    ok: bool
    error: str | None
    message: str | None
    stop_reason: str | None
    events: Annotated[list[dict], operator.add]


class AgentRuntime(Protocol):
    service_unavailable_message: str

    def make_step(self, step_type: str, title: str, body: str, *, meta: dict | None = None) -> dict:
        ...

    def validate_image(self, image_path: Path) -> bool:
        ...

    def classify_endoscopy(self, image_path: Path) -> dict:
        ...

    def build_local_classification_fallback(self, reason: str) -> dict:
        ...

    def enhance_input_image(self, input_image_path: str, output_dir: str) -> dict:
        ...

    def diagnose_original_image(self, input_image_path: str, output_dir: str, lesion_type: str | None = None) -> dict:
        ...

    def annotate_enhanced_image(self, enhancement_result: dict, lesion_result: dict, output_dir: str) -> dict:
        ...

    def diagnose_enhanced_image(self, enhancement_result: dict, output_dir: str, lesion_type: str | None = None) -> dict:
        ...

    def process_image(self, input_image_path: str, output_dir: str, lesion_type: str | None = None) -> dict:
        ...

    def build_result_image_url(self, run_id: str, upload_path: Path, result_image_path: Path) -> str:
        ...

    def generate_lesion_analysis(self, image_path: Path, vision_result: dict) -> dict:
        ...

    def select_report_knowledge(self, lesion_analysis: dict) -> list[dict]:
        ...

    def generate_diagnostic_report(
        self,
        image_path: Path,
        vision_result: dict,
        lesion_analysis: dict | None = None,
        knowledge_context: list[dict] | None = None,
        revision_context: dict | None = None,
    ) -> dict:
        ...

    def build_local_diagnostic_report(self, vision_result: dict, reason: str | None = None) -> dict:
        ...

    def format_structured_report(self, report_json: dict) -> str:
        ...

    def generate_result_reflection(self, image_path: Path, vision_result: dict, report: str) -> dict:
        ...

    def build_local_reflection(self, vision_result: dict, reason: str | None = None) -> dict:
        ...

    def generate_multi_image_diagnostic_report(
        self,
        processed_images: list[dict],
        lesion_type: str,
        revision_context: dict | None = None,
    ) -> dict:
        ...

    def build_local_multi_image_report(self, processed_images: list[dict], reason: str | None = None) -> dict:
        ...

    def generate_multi_result_reflection(self, processed_images: list[dict], report: str) -> dict:
        ...

    def build_local_multi_reflection(self, processed_images: list[dict], reason: str | None = None) -> dict:
        ...

    def answer_followup(self, context: dict, user_text: str) -> str:
        ...

    def fallback_followup_answer(self, context: dict, user_text: str, reason: str | None = None) -> str:
        ...

    def log_debug(self, message: str) -> None:
        ...

    def log_task(self, task_name: str) -> None:
        ...

    def log_json(self, title: str, data) -> None:
        ...


@dataclass(frozen=True)
class FollowupResult:
    answer: str
    fallback: bool = False


REPORT_REVISION_SCORE_THRESHOLD = 5
REPORT_REVISION_CONFIDENCE_THRESHOLD = 60


class ConsoleLoggingRuntime:
    """Runtime proxy that guarantees agent_workflow debug logs reach the Python console."""

    def __init__(self, runtime: AgentRuntime):
        self._runtime = runtime

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    def log_debug(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [agent_workflow] {message}", flush=True)


def create_initial_state(
    run_id: str,
    upload_path: Path,
    upload_url: str,
    lesion_type: str = "polyp",
    upload_items: list[dict] | None = None,
) -> AgentState:
    normalized_upload_items = upload_items or [
        {
            "index": 1,
            "filename": upload_path.name,
            "upload_path": upload_path,
            "upload_url": upload_url,
        }
    ]
    return {
        "run_id": run_id,
        "upload_path": str(upload_path),
        "upload_url": upload_url,
        "upload_items": normalized_upload_items,
        "upload_paths": [str(item.get("upload_path")) for item in normalized_upload_items],
        "upload_urls": [str(item.get("upload_url")) for item in normalized_upload_items],
        "lesion_type": lesion_type,
        "output_dir": str(upload_path.parent / "outputs"),
        "enhancement_result": None,
        "lesion_result": None,
        "result_image_path": None,
        "result_image_url": None,
        "classification": None,
        "vision_result": None,
        "lesion_analysis": None,
        "knowledge_context": None,
        "report": None,
        "reflection": None,
        "report_revision_count": 0,
        "report_revision_max": 1,
        "processed_images": [],
        "ok": True,
        "error": None,
        "message": None,
        "events": [],
    }


def build_analysis_response(state: AgentState) -> dict:
    response = {
        "ok": state.get("ok", True),
        "run_id": state.get("run_id"),
        "upload_url": state.get("upload_url"),
        "upload_urls": state.get("upload_urls"),
        "lesion_type": state.get("lesion_type"),
        "result_image_url": state.get("result_image_url"),
        "enhancement_result": state.get("enhancement_result"),
        "lesion_result": state.get("lesion_result"),
        "classification": state.get("classification"),
        "vision_result": state.get("vision_result"),
        "lesion_analysis": state.get("lesion_analysis"),
        "knowledge_context": state.get("knowledge_context"),
        "report": state.get("report"),
        "report_json": state.get("report_json"),
        "reflection": state.get("reflection"),
        "report_revision_count": state.get("report_revision_count", 0),
        "processed_images": state.get("processed_images"),
        "steps": state.get("events", []),
    }
    if state.get("message"):
        response["message"] = state["message"]
    if state.get("error"):
        response["error"] = state["error"]
    return response


class EndoscopyAnalysisAgent:
    """LangGraph-backed agent that orchestrates the endoscopy analysis workflow."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = ConsoleLoggingRuntime(runtime)
        self.framework = "langgraph" if StateGraph is not None else "linear-fallback"
        self._graph = self._compile_graph() if StateGraph is not None else None

    def invoke(self, initial_state: AgentState) -> AgentState:
        if self._graph is not None:
            self.runtime.log_debug("[LangGraph] ==================== 智能体调用开始 ====================")
            self.runtime.log_debug("[LangGraph] 检测到可用运行框架: LangGraph 状态图引擎")
            self.runtime.log_debug(f"[LangGraph] 同步调用 invoke()，当前会话 ID: {initial_state.get('run_id')}")
            result = self._graph.invoke(initial_state)
            self.runtime.log_debug("[LangGraph] ==================== 智能体调用结束 ====================")
            return result

        self.runtime.log_debug("[Linear-Fallback] ==================== 降级流程开始 ====================")
        self.runtime.log_debug("[Linear-Fallback] 未检测到 LangGraph 依赖，启用弹性降级运行方案")
        self.runtime.log_debug(f"[Linear-Fallback] 执行模拟同步管道，会话 ID: {initial_state.get('run_id')}")
        state = dict(initial_state)
        for state in self.stream(initial_state):
            pass
        self.runtime.log_debug("[Linear-Fallback] ==================== 降级流程结束 ====================")
        return state

    def stream(self, initial_state: AgentState) -> Iterator[AgentState]:
        if self._graph is not None:
            self.runtime.log_debug("[LangGraph] ==================== 智能体流式调用开始 ====================")
            self.runtime.log_debug("[LangGraph] 检测到可用运行框架: LangGraph 状态图引擎")
            self.runtime.log_debug(f"[LangGraph] 启动流式事件监听，当前会话 ID: {initial_state.get('run_id')}")
            for state in self._graph.stream(initial_state, stream_mode="values"):
                yield state
            self.runtime.log_debug("[LangGraph] ==================== 智能体流式调用结束 ====================")
            return

        self.runtime.log_debug("[Linear-Fallback] ==================== 降级流式调用开始 ====================")
        self.runtime.log_debug("[Linear-Fallback] 未检测到 LangGraph 依赖，启用弹性降级流式管道")
        yield from self._stream_without_langgraph(initial_state)
        self.runtime.log_debug("[Linear-Fallback] ==================== 降级流式调用结束 ====================")

    def _compile_graph(self):
        self.runtime.log_debug("[LangGraph] 正在初始化 StateGraph 有向无环图结构...")
        graph = StateGraph(AgentState)
        
        self.runtime.log_debug("[LangGraph] 正在配置状态图节点 (Nodes)...")
        graph.add_node("receive_upload", self._receive_upload)
        graph.add_node("batch_process_images", self._batch_process_images)
        graph.add_node("generate_multi_report", self._generate_multi_report)
        graph.add_node("validate_image", self._validate_image)
        graph.add_node("classify_image", self._classify_image)
        graph.add_node("run_lesion_tools", self._run_lesion_tools)
        graph.add_node("run_iretinex_enhancement", self._run_iretinex_enhancement)
        graph.add_node("annotate_result_image", self._annotate_result_image)
        graph.add_node("extract_lesion_analysis", self._extract_lesion_analysis)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("regenerate_report", self._regenerate_report)
        graph.add_node("regenerate_multi_report", self._regenerate_multi_report)
        graph.add_node("reflect_result", self._reflect_result)
        graph.add_node("publish_result_image", self._publish_result_image)

        self.runtime.log_debug("[LangGraph] 正在配置状态图静态边与条件边 (Edges & Conditional Edges)...")
        graph.add_edge(START, "receive_upload")
        graph.add_edge("receive_upload", "batch_process_images")
        graph.add_conditional_edges(
            "batch_process_images",
            self._route_after_vision,
            {"report": "retrieve_knowledge", "end": END},
        )
        graph.add_edge("retrieve_knowledge", "generate_multi_report")
        graph.add_edge("generate_multi_report", "reflect_result")
        graph.add_conditional_edges(
            "validate_image",
            self._route_after_validation,
            {"classify": "classify_image", "end": END},
        )
        graph.add_conditional_edges(
            "classify_image",
            self._route_after_classification,
            {"vision": "run_lesion_tools", "end": END},
        )
        graph.add_conditional_edges(
            "run_lesion_tools",
            self._route_after_vision,
            {"report": "run_iretinex_enhancement", "end": END},
        )
        graph.add_conditional_edges(
            "run_iretinex_enhancement",
            self._route_after_vision,
            {"report": "annotate_result_image", "end": END},
        )
        graph.add_conditional_edges(
            "annotate_result_image",
            self._route_after_vision,
            {"report": "extract_lesion_analysis", "end": END},
        )
        graph.add_edge("extract_lesion_analysis", "retrieve_knowledge")
        graph.add_edge("generate_report", "reflect_result")
        graph.add_edge("regenerate_report", "publish_result_image")
        graph.add_edge("regenerate_multi_report", "publish_result_image")
        graph.add_conditional_edges(
            "reflect_result",
            self._route_after_reflection,
            {
                "regenerate_report": "regenerate_report",
                "regenerate_multi_report": "regenerate_multi_report",
                "publish": "publish_result_image",
            },
        )
        graph.add_edge("publish_result_image", END)
        
        self.runtime.log_debug("[LangGraph] StateGraph 拓扑结构构建完毕，正在编译引擎...")
        compiled_graph = graph.compile()
        self.runtime.log_debug("[LangGraph] StateGraph 编译成功！图引擎已就绪。")
        return compiled_graph

    def _stream_without_langgraph(self, initial_state: AgentState) -> Iterator[AgentState]:
        self.runtime.log_debug("[Linear-Fallback] 线性模拟引擎激活，开始依次步进执行业务节点")
        state = dict(initial_state)
        state = _merge_state(state, self._receive_upload(state))
        yield state

        for node in (
            self._batch_process_images,
            self._retrieve_knowledge,
            self._generate_multi_report,
            self._reflect_result,
        ):
            self.runtime.log_debug(f"[Linear-Fallback] [串行管道] 执行节点: {node.__name__}")
            state = _merge_state(state, node(state))
            yield state
            if node.__name__ == "_batch_process_images" and self._route_after_vision(state) == "end":
                return

        if self._route_after_reflection(state) == "regenerate_multi_report":
            state = _merge_state(state, self._regenerate_multi_report(state))
            yield state

        state = _merge_state(state, self._publish_result_image(state))
        yield state

    def _receive_upload(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        upload_path = Path(state["upload_path"])
        upload_items = state.get("upload_items") or []
        self.runtime.log_debug(f"{prefix} >>> 节点 [receive_upload] 激活运行 <<<")
        self.runtime.log_debug(f"{prefix} [receive_upload] 成功接收图片: {upload_path.name}，正在分发状态")
        if len(upload_items) > 1:
            names = "、".join(str(item.get("filename")) for item in upload_items)
            return {
                "events": [
                    self.runtime.make_step(
                        "upload",
                        "收到多张图片",
                        f"我已接收并保存用户上传的 {len(upload_items)} 张图片：{names}。",
                    )
                ]
            }
        return {
            "events": [
                self.runtime.make_step(
                    "upload",
                    "收到图片",
                    f"我已接收并保存用户上传的图片：{upload_path.name}。",
                )
            ]
        }

    def _validate_image(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [validate_image] 激活运行 <<<")
        upload_path = Path(state["upload_path"])
        if not self.runtime.validate_image(upload_path):
            self.runtime.log_debug(f"{prefix} [validate_image] 图像合规性校验失败！文件无法识别。")
            return {
                "ok": False,
                "error": "上传文件不是有效图像。",
                "stop_reason": "invalid_image",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "图片校验失败",
                        "上传文件不是有效图像，我无法继续分析。",
                    )
                ],
            }

        self.runtime.log_debug(f"{prefix} [validate_image] 图像合规性校验通过！")
        return {
            "events": [
                self.runtime.make_step(
                    "check",
                    "图片校验通过",
                    "文件格式与图像完整性检查通过，开始进行内镜图像预检。",
                )
            ]
        }

    def _classify_image(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [classify_image] 激活运行 <<<")
        upload_path = Path(state["upload_path"])
        try:
            self.runtime.log_debug(f"{prefix} [classify_image] 正在调起大模型进行图像内镜分类识别...")
            classification = self.runtime.classify_endoscopy(upload_path)
            self.runtime.log_debug(f"{prefix} [classify_image] 分类成功！大模型决策结果: is_endoscopy={classification.get('is_endoscopy')}, reason={classification.get('reason')}")
            warning_event = None
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"{prefix} [classify_image] 内镜预检降级继续本地分析：{technical_message}")
            classification = self.runtime.build_local_classification_fallback(user_message)
            warning_event = self.runtime.make_step("warning", "AI 预检降级", classification["reason"])

        events = []
        if warning_event is not None:
            events.append(warning_event)

        if classification.get("is_endoscopy"):
            events.append(
                self.runtime.make_step(
                    "classification",
                    "AI 预检判断",
                    f"我判断这是一张内镜医学影像。依据：{classification.get('reason') or '图像特征符合内镜视野'}。",
                )
            )
            return {"classification": classification, "events": events}

        message = "这张图片未通过内镜图像预检，未执行 IRetinex 处理。"
        events.append(
            self.runtime.make_step(
                "classification",
                "AI 预检判断",
                f"我判断这不是内镜医学影像，因此停止本地病灶定位流程。依据：{classification.get('reason') or '缺少典型内镜图像特征'}。",
            )
        )
        return {
            "classification": classification,
            "message": message,
            "stop_reason": "not_endoscopy",
            "events": events,
        }

    def _build_image_meta(self, state: AgentState, upload_path: Path, images: list[tuple[str, Path]]) -> dict:
        image_items = []
        for caption, image_path in images:
            resolved_path = image_path.resolve()
            if not resolved_path.is_file():
                continue
            image_items.append(
                {
                    "url": self.runtime.build_result_image_url(state["run_id"], upload_path, resolved_path),
                    "caption": caption,
                }
            )
        if not image_items:
            return {}
        return {
            "image_url": image_items[0]["url"],
            "image_caption": image_items[0]["caption"],
            "image_urls": image_items,
        }

    def _batch_process_one_image(self, state: AgentState, item: dict) -> dict:
        run_id = state["run_id"]
        lesion_type = state.get("lesion_type") or "polyp"
        upload_path = Path(item["upload_path"])
        output_dir = upload_path.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        events = []

        if not self.runtime.validate_image(upload_path):
            return {
                "ok": False,
                "error": "上传文件不是有效图像。",
                "stop_reason": "invalid_image",
                "events": [
                    self.runtime.make_step("error", "图片校验失败", f"{item['filename']} 不是有效图像，已跳过这张图片。")
                ],
            }
        events.append(self.runtime.make_step("check", "图片校验通过", f"{item['filename']} 文件格式与图像完整性检查通过。"))

        try:
            classification = self.runtime.classify_endoscopy(upload_path)
            warning_event = None
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"batch image classification fallback: {technical_message}")
            classification = self.runtime.build_local_classification_fallback(user_message)
            warning_event = self.runtime.make_step("warning", "AI 预检降级", classification["reason"])

        if warning_event is not None:
            events.append(warning_event)

        if not classification.get("is_endoscopy"):
            events.append(
                self.runtime.make_step(
                    "classification",
                    "AI 预检判断",
                    f"{item['filename']} 未通过内镜医学图像预检，已跳过本地视觉处理。依据：{classification.get('reason') or '缺少典型内镜图像特征'}。",
                )
            )
            return {
                "ok": False,
                "classification": classification,
                "error": "图片未通过内镜图像预检。",
                "stop_reason": "not_endoscopy",
                "events": events,
            }

        events.append(
            self.runtime.make_step(
                "classification",
                "AI 预检判断",
                f"{item['filename']} 通过内镜医学图像预检。依据：{classification.get('reason') or '图像特征符合内镜视野'}。",
            )
        )

        try:
            lesion_result = self.runtime.diagnose_original_image(str(upload_path), str(output_dir), lesion_type)
        except Exception as exc:
            return {
                "ok": False,
                "classification": classification,
                "error": "本地病灶定位失败。",
                "stop_reason": "lesion_error",
                "events": events + [self.runtime.make_step("error", "病灶定位失败", f"{item['filename']} 的 YOLO/MedSAM 处理失败：{exc}")],
            }
        if lesion_result.get("status") != "success":
            return {
                "ok": False,
                "classification": classification,
                "lesion_result": lesion_result,
                "error": "本地病灶定位失败。",
                "stop_reason": "lesion_error",
                "events": events + [self.runtime.make_step("error", "本地模型返回失败", f"{item['filename']} 的病灶定位失败：{lesion_result.get('message', '未知错误')}")],
            }

        lesion_image_path = lesion_result.get("lesion_image_path")
        lesion_body = (
            f"YOLO/MedSAM 已在 {item['filename']} 上完成病灶定位，检测框坐标：{lesion_result.get('bbox')}。"
            if lesion_result.get("polyp_detected")
            else f"YOLO/MedSAM 已在 {item['filename']} 上完成检查，未检测到明显疑似病灶区域。"
        )
        events.append(self.runtime.make_step("vision", "病灶位置标注完成", lesion_body))

        try:
            enhancement_result = self.runtime.enhance_input_image(str(upload_path), str(output_dir))
        except Exception as exc:
            return {
                "ok": False,
                "classification": classification,
                "lesion_result": lesion_result,
                "error": "IRetinex 增强失败。",
                "stop_reason": "enhancement_error",
                "events": events + [self.runtime.make_step("error", "IRetinex 增强失败", f"{item['filename']} 增强失败：{exc}")],
            }
        fusion_path = Path(enhancement_result.get("fusion_path") or "")
        if enhancement_result.get("status") != "success" or not fusion_path.is_file():
            return {
                "ok": False,
                "classification": classification,
                "lesion_result": lesion_result,
                "enhancement_result": enhancement_result,
                "error": "IRetinex 增强失败。",
                "stop_reason": "enhancement_error",
                "events": events + [self.runtime.make_step("error", "IRetinex 增强失败", f"{item['filename']} 未生成可读取增强图。")],
            }

        events.append(self.runtime.make_step("enhancement", "IRetinex 增强完成", f"{item['filename']} 已完成 IRetinex 增强。"))

        try:
            vision_result = self.runtime.annotate_enhanced_image(enhancement_result, lesion_result, str(output_dir))
        except Exception as exc:
            return {
                "ok": False,
                "classification": classification,
                "lesion_result": lesion_result,
                "enhancement_result": enhancement_result,
                "error": "本地病灶标注失败。",
                "stop_reason": "annotation_error",
                "events": events + [self.runtime.make_step("error", "病灶标注失败", f"{item['filename']} 增强图标注失败：{exc}")],
            }
        result_image_path = Path(vision_result.get("result_image_path") or "").resolve()
        if vision_result.get("status") != "success" or not result_image_path.is_file():
            return {
                "ok": False,
                "classification": classification,
                "lesion_result": lesion_result,
                "enhancement_result": enhancement_result,
                "vision_result": vision_result,
                "error": "本地病灶标注失败。",
                "stop_reason": "annotation_error",
                "events": events + [self.runtime.make_step("error", "本地标注返回失败", f"{item['filename']} 未生成可读取结果图。")],
            }

        result_image_url = self.runtime.build_result_image_url(run_id, upload_path, result_image_path)
        result_body = (
            f"已将 {item['filename']} 原图中的病灶坐标和分割区域标注到 IRetinex 增强图上。"
            if vision_result.get("polyp_detected")
            else f"已生成 {item['filename']} 的 IRetinex 增强图；未检测到明显疑似病灶区域。"
        )
        comparison_images = [
            {
                "url": item.get("upload_url"),
                "caption": "原图",
            }
        ]
        if lesion_image_path and Path(lesion_image_path).is_file():
            comparison_images.append(
                {
                    "url": self.runtime.build_result_image_url(run_id, upload_path, Path(lesion_image_path).resolve()),
                    "caption": "病灶标记图",
                }
            )
        comparison_images.extend(
            [
                {
                    "url": self.runtime.build_result_image_url(run_id, upload_path, fusion_path.resolve()),
                    "caption": "IRetinex 增强图",
                },
                {
                    "url": result_image_url,
                    "caption": "增强后标记图",
                },
            ]
        )
        events.append(
            self.runtime.make_step(
                "vision",
                "增强图病灶标注完成",
                result_body,
                meta={
                    "comparison_title": f"处理结果 - 图片 {item['index']}",
                    "comparison_images": comparison_images,
                    "image_url": result_image_url,
                    "image_caption": f"图片 {item['index']} 增强图病灶标注结果",
                },
            )
        )

        try:
            lesion_analysis = self.runtime.generate_lesion_analysis(result_image_path, vision_result)
            lesion_type_text = lesion_analysis.get("lesion_type") or "未知"
            confidence = lesion_analysis.get("confidence_level") or "未标注"
            lesion_count = lesion_analysis.get("lesion_count")
            events.append(
                self.runtime.make_step(
                    "analysis",
                    "结构化病灶分析完成",
                    f"{item['filename']} 已输出结构化病灶信息：lesion_type={lesion_type_text}，confidence_level={confidence}，lesion_count={lesion_count}。",
                )
            )
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"batch image lesion analysis fallback: {technical_message}")
            lesion_analysis = None
            events.append(
                self.runtime.make_step(
                    "warning",
                    "结构化病灶分析降级",
                    f"{item['filename']} 的结构化病灶分析暂时不可用，后续报告将仅基于本地视觉结果汇总。原因：{user_message}",
                )
            )

        return {
            "ok": True,
            "classification": classification,
            "lesion_result": lesion_result,
            "enhancement_result": enhancement_result,
            "vision_result": vision_result,
            "lesion_analysis": lesion_analysis,
            "result_image_path": str(result_image_path),
            "result_image_url": result_image_url,
            "events": events,
        }

    def _batch_process_images(self, state: AgentState) -> AgentState:
        upload_items = state.get("upload_items") or []
        all_events = []
        processed_images = []
        for item in upload_items:
            all_events.append(
                self.runtime.make_step(
                    "batch",
                    f"开始处理图片 {item['index']}/{len(upload_items)}",
                    f"正在单独处理：{item['filename']}。YOLO/MedSAM 与 IRetinex 会分别在这张图上运行。",
                )
            )
            image_state = self._batch_process_one_image(state, item)
            for step in image_state.get("events", []):
                step = dict(step)
                step["title"] = f"图片 {item['index']} - {step.get('title', '分析步骤')}"
                all_events.append(step)
            image_response = {
                "ok": image_state.get("ok", True),
                "error": image_state.get("error"),
                "lesion_type": state.get("lesion_type"),
                "upload_url": item.get("upload_url"),
                "classification": image_state.get("classification"),
                "enhancement_result": image_state.get("enhancement_result"),
                "lesion_result": image_state.get("lesion_result"),
                "vision_result": image_state.get("vision_result"),
                "lesion_analysis": image_state.get("lesion_analysis"),
                "result_image_url": image_state.get("result_image_url"),
            }
            processed_images.append(
                {
                    "index": item["index"],
                    "filename": item["filename"],
                    "upload_path": str(item["upload_path"]),
                    "upload_url": item.get("upload_url"),
                    "result_image_path": image_state.get("result_image_path"),
                    "result_image_url": image_state.get("result_image_url"),
                    "ok": image_state.get("ok", True),
                    "error": image_state.get("error"),
                    "state": image_response,
                }
            )

        ok_images = [item for item in processed_images if item.get("ok", True) and item.get("result_image_path")]
        if not ok_images:
            return {
                "ok": False,
                "error": "所有图片都未能完成本地处理。",
                "stop_reason": "vision_error",
                "processed_images": processed_images,
                "events": all_events,
            }

        last_result = next((item for item in reversed(ok_images) if item.get("result_image_url")), None)
        last_state = (last_result or {}).get("state") or {}
        return {
            "processed_images": processed_images,
            "result_image_path": last_result.get("result_image_path") if last_result else None,
            "result_image_url": last_result.get("result_image_url") if last_result else None,
            "classification": last_state.get("classification"),
            "enhancement_result": last_state.get("enhancement_result"),
            "lesion_result": last_state.get("lesion_result"),
            "lesion_analysis": last_state.get("lesion_analysis"),
            "vision_result": {"status": "success", "image_count": len(ok_images), "processed_images": ok_images},
            "events": all_events,
        }

    def _generate_multi_report(self, state: AgentState) -> AgentState:
        processed_images = [
            item
            for item in (state.get("processed_images") or [])
            if item.get("ok", True) and item.get("result_image_path")
        ]
        lesion_type = state.get("lesion_type") or "polyp"
        revision_context = state.get("reflection") if state.get("report_revision_count", 0) else None
        image_count = len(processed_images)
        try:
            report_payload = self.runtime.generate_multi_image_diagnostic_report(processed_images, lesion_type, revision_context)
            report = self.runtime.format_structured_report(report_payload) if hasattr(self.runtime, "format_structured_report") else report_payload
            title = "辅助报告已根据反思修订" if revision_context else "辅助诊断报告生成完成"
            return {
                "report": report,
                "report_json": report_payload if isinstance(report_payload, dict) else None,
                "events": [self.runtime.make_step("report", title, report, meta={"report_json": report_payload} if isinstance(report_payload, dict) else None)],
            }
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"multi-image report fallback: {technical_message}")
            report_payload = self.runtime.build_local_multi_image_report(processed_images, user_message)
            report = self.runtime.format_structured_report(report_payload) if hasattr(self.runtime, "format_structured_report") else report_payload
            return {
                "report": report,
                "report_json": report_payload if isinstance(report_payload, dict) else None,
                "events": [
                    self.runtime.make_step("warning", "综合报告降级", f"外部 AI 综合分析暂时不可用，已根据 {image_count} 张成功图片生成本地汇总。原因：{user_message}"),
                    self.runtime.make_step("report", "辅助诊断报告生成完成", report, meta={"report_json": report_payload} if isinstance(report_payload, dict) else None),
                ],
            }

    def _run_iretinex_enhancement(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> node [run_iretinex_enhancement] activated <<<")
        upload_path = Path(state["upload_path"])
        output_dir = Path(state["output_dir"])
        lesion_type = state.get("lesion_type") or "polyp"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.runtime.log_task(f"{prefix} run IRetinex enhancement")
            enhancement_result = self.runtime.enhance_input_image(str(upload_path), str(output_dir))
            self.runtime.log_json(f"{prefix} IRetinex enhancement result", enhancement_result)
        except Exception as exc:
            self.runtime.log_debug(f"{prefix} [run_iretinex_enhancement] failed: {exc}")
            return {
                "ok": False,
                "error": "IRetinex 增强失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "IRetinex 增强失败",
                        "本地图像增强步骤执行失败，暂时无法继续分析。",
                    )
                ],
            }

        fusion_path = Path(enhancement_result.get("fusion_path") or "")
        if enhancement_result.get("status") != "success" or not fusion_path.is_file():
            return {
                "ok": False,
                "enhancement_result": enhancement_result,
                "error": "IRetinex 增强失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "IRetinex 增强失败",
                        f"本地图像增强步骤未生成可读取图像。返回信息：{enhancement_result.get('message', '未知错误')}",
                    )
                ],
            }

        body = (
            "IRetinex 已完成原图增强。"
            f"信息熵：{enhancement_result.get('entropy_before'):.4f} -> {enhancement_result.get('entropy_after'):.4f}；"
            f"处理耗时：{enhancement_result.get('processing_time_ms')} ms。"
        )
        return {
            "enhancement_result": enhancement_result,
            "events": [self.runtime.make_step("enhancement", "IRetinex 增强完成", body)],
        }

    def _run_lesion_tools(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> node [run_lesion_tools] activated <<<")
        upload_path = Path(state["upload_path"])
        output_dir = Path(state["output_dir"])
        lesion_type = state.get("lesion_type") or "polyp"

        try:
            self.runtime.log_task(f"{prefix} run YOLO/MedSAM lesion tools on original image ({lesion_type})")
            lesion_result = self.runtime.diagnose_original_image(str(upload_path), str(output_dir), lesion_type)
            self.runtime.log_json(f"{prefix} original-image lesion tool result", lesion_result)
        except Exception as exc:
            self.runtime.log_debug(f"{prefix} [run_lesion_tools] failed: {exc}")
            return {
                "ok": False,
                "error": "本地病灶定位失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "病灶定位失败",
                        "本地检测或分割步骤执行失败，请稍后重试。",
                    )
                ],
            }

        if lesion_result.get("status") != "success":
            return {
                "ok": False,
                "lesion_result": lesion_result,
                "error": "本地病灶定位失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "本地模型返回失败",
                        f"本地病灶定位模型处理原图失败。返回信息：{lesion_result.get('message', '未知错误')}",
                    )
                ],
            }

        if lesion_result.get("polyp_detected"):
            body = (
                "YOLO/MedSAM 已直接基于原图完成病灶定位，并在原图上标出疑似病灶区域。"
                f"检测框坐标：{lesion_result.get('bbox')}。"
            )
        else:
            body = "YOLO/MedSAM 已直接基于原图完成检查，未检测到明显疑似病灶区域。"

        lesion_image_path = lesion_result.get("lesion_image_path")
        if lesion_image_path:
            resolved_lesion_image = Path(lesion_image_path).resolve()
            self.runtime.log_debug(
                f"{prefix} [run_lesion_tools] 准备在第4步显示原图病灶位置标注图: "
                f"path={resolved_lesion_image}, exists={resolved_lesion_image.is_file()}"
            )
            if resolved_lesion_image.is_file():
                image_url = self.runtime.build_result_image_url(state["run_id"], upload_path, resolved_lesion_image)
                self.runtime.log_debug(f"{prefix} [run_lesion_tools] 原图病灶位置标注图 URL: {image_url}")
        else:
            self.runtime.log_debug(f"{prefix} [run_lesion_tools] 第4步没有可显示的原图病灶位置标注图路径。")

        return {
            "lesion_result": lesion_result,
            "events": [self.runtime.make_step("vision", "病灶位置标注完成", body)],
        }

    def _annotate_result_image(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> node [annotate_result_image] activated <<<")
        upload_path = Path(state["upload_path"])
        output_dir = Path(state["output_dir"])
        enhancement_result = state.get("enhancement_result") or {}
        lesion_result = state.get("lesion_result") or {}

        try:
            self.runtime.log_task(f"{prefix} mark original-image lesion result on enhanced image")
            vision_result = self.runtime.annotate_enhanced_image(enhancement_result, lesion_result, str(output_dir))
            self.runtime.log_json(f"{prefix} enhanced-image annotation result", vision_result)
        except Exception as exc:
            self.runtime.log_debug(f"{prefix} [annotate_result_image] failed: {exc}")
            return {
                "ok": False,
                "enhancement_result": enhancement_result,
                "lesion_result": lesion_result,
                "error": "本地病灶标注失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "病灶标注失败",
                        "本地病灶标注步骤执行失败，请稍后重试。",
                    )
                ],
            }

        if vision_result.get("status") != "success" or not vision_result.get("result_image_path"):
            return {
                "ok": False,
                "enhancement_result": enhancement_result,
                "lesion_result": lesion_result,
                "vision_result": vision_result,
                "error": "本地病灶标注失败。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "本地标注返回失败",
                        f"本地模型未生成可读取的结果图。返回信息：{vision_result.get('message', '未知错误')}",
                    )
                ],
            }

        result_image_path = Path(vision_result["result_image_path"]).resolve()
        if result_image_path.is_dir() or not result_image_path.is_file():
            return {
                "ok": False,
                "enhancement_result": enhancement_result,
                "lesion_result": lesion_result,
                "vision_result": vision_result,
                "error": "本地结果图不可读取。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "结果图不可用",
                        f"本地模型返回的结果图路径不可读取：{result_image_path}",
                    )
                ],
            }

        result_image_url = self.runtime.build_result_image_url(
            state["run_id"],
            upload_path,
            result_image_path,
        )
        self.runtime.log_debug(
            f"{prefix} [annotate_result_image] 第6步增强图病灶标注结果将派发到前端显示: "
            f"path={result_image_path}, url={result_image_url}"
        )
        if vision_result.get("polyp_detected"):
            body = (
                "已将原图中得到的病灶坐标和分割区域标注到 iRetinex 增强图上。"
                f"原图检测框坐标：{vision_result.get('bbox')}。"
            )
        else:
            body = "已生成 iRetinex 增强图；YOLO/MedSAM 在原图中未检测到明显疑似病灶区域。"

        comparison_images = [
            {
                "url": state.get("upload_url"),
                "caption": "原图",
            }
        ]
        lesion_image_path = lesion_result.get("lesion_image_path")
        if lesion_image_path and Path(lesion_image_path).is_file():
            comparison_images.append(
                {
                    "url": self.runtime.build_result_image_url(state["run_id"], upload_path, Path(lesion_image_path).resolve()),
                    "caption": "病灶标记图",
                }
            )
        fusion_path = Path(enhancement_result.get("fusion_path") or "")
        if fusion_path.is_file():
            comparison_images.append(
                {
                    "url": self.runtime.build_result_image_url(state["run_id"], upload_path, fusion_path.resolve()),
                    "caption": "IRetinex 增强图",
                }
            )
        comparison_images.append(
            {
                "url": result_image_url,
                "caption": "增强后标记图",
            }
        )

        return {
            "enhancement_result": enhancement_result,
            "lesion_result": lesion_result,
            "vision_result": vision_result,
            "result_image_path": str(result_image_path),
            "result_image_url": result_image_url,
            "events": [
                self.runtime.make_step(
                    "vision",
                    "增强图病灶标注完成",
                    body,
                    meta={
                        "comparison_title": "处理结果",
                        "comparison_images": comparison_images,
                        "image_url": result_image_url,
                        "image_caption": "增强图病灶标注结果",
                    },
                )
            ],
        }

    def _run_vision_tools(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [run_vision_tools] 激活运行 <<<")
        upload_path = Path(state["upload_path"])
        output_dir = Path(state["output_dir"])
        lesion_type = state.get("lesion_type") or "polyp"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.runtime.log_task(f"{prefix} 智能体调用 IRetinex/YOLO/MedSAM 工具")
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 视觉工具输入图片：{upload_path}")
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 视觉工具输出目录：{output_dir}")
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 开始调度底层神经网络运算...")
            vision_result = self.runtime.process_image(str(upload_path), str(output_dir), lesion_type)
            self.runtime.log_json(f"{prefix} 视觉工具返回原始数据", vision_result)
        except Exception as exc:
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 本地视觉工具执行报错：{exc}")
            return {
                "ok": False,
                "error": "本地图像处理失败，请稍后重试。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "本地模型执行失败",
                        "IRetinex 增强或病灶定位流程执行失败，请稍后重试。",
                    )
                ],
            }

        if vision_result.get("status") != "success" or not vision_result.get("result_image_path"):
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 本地模型状态返回失败")
            return {
                "ok": False,
                "vision_result": vision_result,
                "error": "本地图像处理失败，请稍后重试。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "本地模型返回失败",
                        f"本地模型没有生成可展示的结果图。返回信息：{vision_result.get('message', '未知错误')}",
                    )
                ],
            }

        result_image_path = Path(vision_result["result_image_path"]).resolve()
        if result_image_path.is_dir() or not result_image_path.is_file():
            self.runtime.log_debug(f"{prefix} [run_vision_tools] 结果图像物理路径不可达: {result_image_path}")
            return {
                "ok": False,
                "vision_result": vision_result,
                "error": "本地处理结果图片不可读取。",
                "stop_reason": "vision_error",
                "events": [
                    self.runtime.make_step(
                        "error",
                        "结果图不可用",
                        f"本地模型返回的结果图路径不可读取：{result_image_path}",
                    )
                ],
            }

        self.runtime.log_debug(f"{prefix} [run_vision_tools] 本地算法处理圆满成功。")
        self.runtime.log_debug(f"{prefix} [run_vision_tools] 息肉病灶识别状态: {vision_result.get('polyp_detected')}")
        self.runtime.log_debug(f"{prefix} [run_vision_tools] 检测坐标框: {vision_result.get('bbox')}")
        self.runtime.log_debug(f"{prefix} [run_vision_tools] 处理用时/运行设备: {vision_result.get('device')}")

        result_image_url = self.runtime.build_result_image_url(
            state["run_id"],
            upload_path,
            result_image_path,
        )
        if vision_result.get("polyp_detected"):
            body = (
                "我已生成病灶位置标注图，并标记到疑似病灶区域。"
                f"检测框坐标：{vision_result.get('bbox')}。"
            )
        else:
            body = "我已生成增强后的结果图。本地模型未检测到明显病灶区域。"

        return {
            "vision_result": vision_result,
            "result_image_path": str(result_image_path),
            "result_image_url": result_image_url,
            "events": [self.runtime.make_step("vision", "IRetinex 与病灶定位完成", body)],
        }

    def _generate_report(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [generate_report] 激活运行 <<<")
        result_image_path = Path(state["result_image_path"])
        vision_result = state.get("vision_result") or {}
        lesion_analysis = state.get("lesion_analysis")
        knowledge_context = state.get("knowledge_context")
        revision_context = state.get("reflection") if state.get("report_revision_count", 0) else None
        try:
            self.runtime.log_debug(f"{prefix} [generate_report] 正在调度医学辅助模型生成诊断报告文本...")
            report_payload = self.runtime.generate_diagnostic_report(
                result_image_path,
                vision_result,
                lesion_analysis,
                knowledge_context,
                revision_context,
            )
            report = self.runtime.format_structured_report(report_payload) if hasattr(self.runtime, "format_structured_report") else report_payload
            self.runtime.log_debug(f"{prefix} [generate_report] 报告文本生成成功！长度: {len(report)} 字")
            title = "根据结果评估，判断需要修订诊断报告" if revision_context else "诊断报告"
            return {
                "report": report,
                "report_json": report_payload if isinstance(report_payload, dict) else None,
                "events": [self.runtime.make_step("report", title, report, meta={"report_json": report_payload} if isinstance(report_payload, dict) else None)],
            }
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"{prefix} [generate_report] 报告生成触发降级策略：{technical_message}")
            report_payload = self.runtime.build_local_diagnostic_report(vision_result, user_message)
            report = self.runtime.format_structured_report(report_payload) if hasattr(self.runtime, "format_structured_report") else report_payload
            return {
                "report": report,
                "report_json": report_payload if isinstance(report_payload, dict) else None,
                "events": [
                    self.runtime.make_step(
                        "warning",
                        "报告生成降级",
                        "外部 AI 报告生成暂时不可用，已根据本地模型结果生成保底报告。",
                    ),
                    self.runtime.make_step("report", "辅助报告生成完成", report, meta={"report_json": report_payload} if isinstance(report_payload, dict) else None),
                ],
            }

    def _regenerate_report(self, state: AgentState) -> AgentState:
        revision_count = int(state.get("report_revision_count") or 0) + 1
        revision_state = dict(state)
        revision_state["report_revision_count"] = revision_count
        update = self._generate_report(revision_state)
        update["report_revision_count"] = revision_count
        return update

    def _regenerate_multi_report(self, state: AgentState) -> AgentState:
        revision_count = int(state.get("report_revision_count") or 0) + 1
        revision_state = dict(state)
        revision_state["report_revision_count"] = revision_count
        update = self._generate_multi_report(revision_state)
        update["report_revision_count"] = revision_count
        return update

    def _extract_lesion_analysis(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [extract_lesion_analysis] 激活运行 <<<")
        result_image_path = Path(state["result_image_path"])
        vision_result = state.get("vision_result") or {}
        try:
            self.runtime.log_debug(f"{prefix} [extract_lesion_analysis] 正在请求大模型输出结构化病灶 JSON...")
            lesion_analysis = self.runtime.generate_lesion_analysis(result_image_path, vision_result)
            self.runtime.log_json(f"{prefix} structured lesion analysis", lesion_analysis)
            lesion_type = lesion_analysis.get("lesion_type") or "未知"
            confidence = lesion_analysis.get("confidence_level") or "未标注"
            lesion_count = lesion_analysis.get("lesion_count")
            return {
                "lesion_analysis": lesion_analysis,
                "events": [
                    self.runtime.make_step(
                        "analysis",
                        "结构化病灶分析完成",
                        f"大模型已输出病灶结构化信息：lesion_type={lesion_type}，confidence_level={confidence}，lesion_count={lesion_count}。",
                    )
                ],
            }
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"{prefix} [extract_lesion_analysis] 结构化分析失败，跳过知识库检索：{technical_message}")
            return {
                "lesion_analysis": None,
                "knowledge_context": [],
                "events": [
                    self.runtime.make_step(
                        "warning",
                        "结构化病灶分析降级",
                        f"大模型结构化病灶分析暂时不可用，将不使用知识库检索辅助报告。原因：{user_message}",
                    )
                ],
            }

    def _retrieve_knowledge(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [retrieve_knowledge] 激活运行 <<<")
        all_processed_images = state.get("processed_images") or []
        processed_images = [
            item
            for item in all_processed_images
            if item.get("ok", True) and item.get("result_image_path")
        ]
        if processed_images:
            knowledge_context = []
            seen = set()
            for item in processed_images:
                image_state = item.get("state") or {}
                lesion_analysis = image_state.get("lesion_analysis")
                if not lesion_analysis:
                    continue
                try:
                    image_knowledge = self.runtime.select_report_knowledge(lesion_analysis)
                except Exception as exc:
                    self.runtime.log_debug(
                        f"{prefix} [retrieve_knowledge] 图片 {item.get('index')} 知识库检索失败，继续处理其它图片：{exc}"
                    )
                    image_knowledge = []
                image_state["knowledge_context"] = image_knowledge
                for rule in image_knowledge:
                    key = (
                        str(rule.get("disease_type", "")),
                        str(rule.get("rule_id", "")),
                        str(rule.get("source_file", "")),
                        str(rule.get("source_section_or_page", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    knowledge_context.append(rule)

            if not knowledge_context:
                return {
                    "knowledge_context": [],
                    "events": [
                        self.runtime.make_step(
                            "knowledge",
                            "知识库检索完成",
                            "未检索到足够相关的本地知识库规则，本次报告将基于图像、病灶标注和结构化分析生成。",
                        )
                    ],
                }

            names = "、".join(
                str(rule.get("cn_name") or rule.get("rule_id") or "未命名规则")
                for rule in knowledge_context[:3]
            )
            return {
                "processed_images": all_processed_images,
                "knowledge_context": knowledge_context,
                "events": [
                    self.runtime.make_step(
                        "knowledge",
                        "知识库检索完成",
                        f"已基于 {len(processed_images)} 张成功图片检索到 {len(knowledge_context)} 条相关规则：{names}。",
                    )
                ],
            }

        lesion_analysis = state.get("lesion_analysis") or {}
        if not lesion_analysis:
            self.runtime.log_debug(f"{prefix} [retrieve_knowledge] 无结构化病灶结果，跳过知识库检索。")
            return {"knowledge_context": []}

        try:
            knowledge_context = self.runtime.select_report_knowledge(lesion_analysis)
            self.runtime.log_json(f"{prefix} selected knowledge context", knowledge_context)
        except Exception as exc:
            self.runtime.log_debug(f"{prefix} [retrieve_knowledge] 知识库检索失败，继续无知识库报告：{exc}")
            return {
                "knowledge_context": [],
                "events": [
                    self.runtime.make_step(
                        "warning",
                        "知识库检索降级",
                        "本地知识库检索暂时不可用，本次报告将仅基于图像、病灶标注和结构化分析生成。",
                    )
                ],
            }

        if not knowledge_context:
            return {
                "knowledge_context": [],
                "events": [
                    self.runtime.make_step(
                        "knowledge",
                        "知识库检索完成",
                        "未检索到足够相关的本地知识库规则，本次报告不附加知识库内容。",
                    )
                ],
            }

        names = "、".join(
            str(rule.get("cn_name") or rule.get("rule_id") or "未命名规则")
            for rule in knowledge_context[:3]
        )
        return {
            "knowledge_context": knowledge_context,
            "events": [
                self.runtime.make_step(
                    "knowledge",
                    "知识库检索完成",
                    f"已从本地知识库检索到 {len(knowledge_context)} 条相关规则：{names}。",
                )
            ],
        }

    def _reflect_result(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [reflect_result] 激活运行 <<<")
        processed_images = [
            item
            for item in (state.get("processed_images") or [])
            if item.get("ok", True) and item.get("result_image_path")
        ]
        vision_result = state.get("vision_result") or {}
        report = state.get("report") or ""
        try:
            self.runtime.log_debug(f"{prefix} [reflect_result] 正在调起质量自检模型反思当前诊断质量...")
            if processed_images:
                reflection = self.runtime.generate_multi_result_reflection(processed_images, report)
            else:
                result_image_path = Path(state["result_image_path"])
                reflection = self.runtime.generate_result_reflection(result_image_path, vision_result, report)
            self.runtime.log_debug(f"{prefix} [reflect_result] 自检完成！置信度: {reflection.get('confidence')}% ({reflection.get('confidence_label')})")
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"{prefix} [reflect_result] 结果自检降级：{technical_message}")
            if processed_images:
                reflection = self.runtime.build_local_multi_reflection(processed_images, user_message)
            else:
                reflection = self.runtime.build_local_reflection(vision_result, user_message)

        confidence_text = "未评估" if reflection.get("confidence") is None else f"{reflection['confidence']}%"
        uncertainty_text = "；".join(reflection.get("uncertainties") or []) or "暂无额外列出"
        body = (
            f"粗略置信度：{confidence_text}（{reflection.get('confidence_label') or '未标注'}）。"
            f"{reflection.get('reflection') or ''}\n"
            f"不确定因素：{uncertainty_text}\n"
            f"{reflection.get('suggestion') or ''}"
        ).strip()
        return {
            "reflection": reflection,
            "events": [self.runtime.make_step("reflection", "结果自检完成", body)],
        }

    def _publish_result_image(self, state: AgentState) -> AgentState:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        self.runtime.log_debug(f"{prefix} >>> 节点 [publish_result_image] 激活运行 <<<")
        self.runtime.log_debug(
            f"{prefix} [publish_result_image] 第9步“生成结果图”已禁用显示，"
            f"不会向前端派发图片。result_image_url={state.get('result_image_url')}"
        )
        return {"events": []}

    def _route_after_receive_upload(self, state: AgentState) -> str:
        upload_items = state.get("upload_items") or []
        route = "batch" if len(upload_items) > 1 else "single"
        self.runtime.log_debug(f"[LangGraph] [route_after_receive_upload]: upload_count={len(upload_items)} -> {route}")
        return route

    def _route_after_validation(self, state: AgentState) -> str:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        reason = state.get("stop_reason")
        route = "end" if reason == "invalid_image" else "classify"
        self.runtime.log_debug(f"{prefix} [条件路由决策] [route_after_validation]: stop_reason={reason} -> 分支跳转至: {route}")
        return route

    def _route_after_classification(self, state: AgentState) -> str:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        reason = state.get("stop_reason")
        route = "end" if reason == "not_endoscopy" else "vision"
        self.runtime.log_debug(f"{prefix} [条件路由决策] [route_after_classification]: stop_reason={reason} -> 分支跳转至: {route}")
        return route

    def _route_after_reflection(self, state: AgentState) -> str:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        revision_count = int(state.get("report_revision_count") or 0)
        revision_max = int(state.get("report_revision_max") or 1)
        if revision_count >= revision_max:
            self.runtime.log_debug(
                f"{prefix} [route_after_reflection]: report revision limit reached "
                f"({revision_count}/{revision_max}) -> publish"
            )
            return "publish"

        reflection = state.get("reflection") or {}
        score = self._coerce_revision_score(reflection)
        confidence = reflection.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = None

        explicit_request = bool(
            reflection.get("revise_report")
            or reflection.get("needs_report_revision")
            or reflection.get("report_revision_required")
        )
        low_confidence = confidence_value is not None and confidence_value < REPORT_REVISION_CONFIDENCE_THRESHOLD
        should_revise = explicit_request or score >= REPORT_REVISION_SCORE_THRESHOLD or low_confidence
        ok_processed_images = [
            item
            for item in (state.get("processed_images") or [])
            if item.get("ok", True) and item.get("result_image_path")
        ]
        has_processed_images = bool(ok_processed_images)
        if should_revise and has_processed_images:
            route = "regenerate_multi_report"
        elif should_revise:
            route = "regenerate_report"
        else:
            route = "publish"
        self.runtime.log_debug(
            f"{prefix} [route_after_reflection]: score={score}, confidence={confidence_value}, "
            f"explicit={explicit_request}, low_confidence={low_confidence}, processed_images={len(ok_processed_images)} -> {route}"
        )
        return route

    @staticmethod
    def _coerce_revision_score(reflection: dict) -> int:
        for key in ("report_revision_score", "revision_score", "rrs"):
            try:
                return max(0, min(10, int(round(float(reflection.get(key))))))
            except (TypeError, ValueError):
                continue
        return 0

    def _route_after_vision(self, state: AgentState) -> str:
        prefix = "[LangGraph]" if self._graph is not None else "[Linear-Fallback]"
        reason = state.get("stop_reason")
        route = "end" if reason == "vision_error" else "report"
        self.runtime.log_debug(f"{prefix} [条件路由决策] [route_after_vision]: stop_reason={reason} -> 分支跳转至: {route}")
        return route


class FollowupAgent:
    """Small agent facade for follow-up dialogue over a completed analysis context."""

    def __init__(self, runtime: AgentRuntime):
        self.runtime = ConsoleLoggingRuntime(runtime)

    def invoke(self, context: dict, user_text: str) -> FollowupResult:
        try:
            return FollowupResult(answer=self.runtime.answer_followup(context, user_text))
        except Exception as exc:
            user_message = getattr(exc, "user_message", str(exc))
            technical_message = getattr(exc, "technical_message", str(exc))
            self.runtime.log_debug(f"多轮追问降级：{technical_message}")
            return FollowupResult(
                answer=self.runtime.fallback_followup_answer(context, user_text, user_message),
                fallback=True,
            )


def _merge_state(base: dict, update: dict) -> AgentState:
    merged = dict(base)
    for key, value in update.items():
        if key == "events":
            merged[key] = [*merged.get(key, []), *value]
        else:
            merged[key] = value
    return merged
