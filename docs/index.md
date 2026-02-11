# BEAM: Token-efficient Multi-agent Inference Framework

**BEAM** 是一个高性能的分布式多智能体推理框架，专注于在保证推理质量的前提下，最大化 Token 利用率与推理效率。

---

## 🚀 核心特性 (Key Features)

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **极致效率**
    ---
    通过智能的 Token 调度算法，显著降低多轮对话中的推理开销。

-   :material-source-branch:{ .lg .middle } **多策略驱动**
    ---
    内置多种搜索与推理策略，支持从简单的并行链到复杂的树状搜索布局。

-   :material-engine:{ .lg .middle } **插件化架构**
    ---
    轻松集成各种主流 LLM API (Qwen, GPT, DeepSeek)，支持自定义智能体逻辑。

-   :material-chart-bar:{ .lg .middle } **深度可视化**
    ---
    提供完整的推理路径可视化工具，让多智能体的决策过程清晰透明。

</div>

---

## 🛠️ 快速上手 (Quick Start)

三步开启 BEAM 推理之旅：

1.  **安装环境**
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置 API 密钥**
    在 `.env` 文件中设置你的模型密钥。

3.  **运行示例**
    ```bash
    python run_pipeline.py --task math_solve
    ```

---

## 📖 导航说明

* 如果你是第一次使用，请查看 [快速开始](getting-started/quickstart.md)。
* 深入了解系统设计，请阅读 [核心架构](getting-started/overview.md)。
* 查阅函数细节，请移步 [API Reference](api-reference/apireference.md)。

---

## 🔗 相关链接

* [:material-github: GitHub 仓库](https://github.com/LJF12342/My_BEAM)
* [:material-bug: 问题反馈](https://github.com/LJF12342/My_BEAM/issues)