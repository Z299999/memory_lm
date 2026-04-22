# w0001-simplex-theory — Simplex Memory Network Theory

LaTeX 理论文档：**Simplex Memory Networks: A Geometric Feedforward Architecture for Coupled Short- and Long-Term Memory**

## 编译方法

```bash
cd writing/w0001-simplex-theory
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

输出：`report.pdf`

## 目录结构

```
w0001-simplex-theory/
├── report.tex          # 主文档
├── report.bib          # 参考文献
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_preliminaries.tex
│   ├── 03_directed_graph.tex
│   ├── 04_motifs.tex
│   ├── 06_interpretation.tex
│   ├── 07_open_directions.tex
│   └── 08_conclusion.tex
└── README.md
```

## 代码实现

完整代码实现在 [exp0414_simplexNet](../../experiments/exp0414_simplexNet/)

- **SMNModule**: 纯 PyTorch nn.Module，用于强化学习/自定义训练
- **SMNFitter**: 高级训练包装器，内置 fit()/predict()/plot()

## 快速链接

| 资源 | 位置 |
|------|------|
| 理论文档 | `writing/w0001-simplex-theory/` |
| 代码实现 | `experiments/exp0414_simplexNet/` |
| RL 示例 | `experiments/exp0414_simplexNet/examples/smn_for_rl.py` |
| API 文档 | `experiments/exp0414_simplexNet/README.md` |
