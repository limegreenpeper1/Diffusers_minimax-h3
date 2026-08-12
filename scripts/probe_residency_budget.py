"""`_residency_requirements_gb()` / `_effective_vram_budget_gb()` の回帰確認。

2026-08-12、予算ガードを直書き定数から実測容量比較へ移した際の検証用
(core/runner.py の `H3_KEEP_TRANSFORMER` ブロック参照)。

確認するのは2点:

1. **判定関数が docs/RESIDENCY.md の収支表と同じ結論を出すこと**。表の各行
   (§「ケース / 必要 / 実効予算」) を構成として入力し、必要量と OK/NG が一致するか見る。
   導出値そのものは ±1GB ずれる (同ドキュメントが明記) ので、必要量は 0.5GB の許容で、
   **判定 (OK/NG) は完全一致**を要求する。
2. この箱の実効予算が実際に何GBになるか (統合メモリ補正込み)。

`H3_KEEP_TRANSFORMER` を立てずに import するので、runner の import 時ガードは走らない
(= このスクリプト自体はどの箱でも読み込める)。モデルは一切ロードしない。

実行: .venv/bin/python -m scripts.probe_residency_budget
"""

import os

# ガードを起動させずに関数だけ使う (既定値のまま import する)。
os.environ.pop("H3_KEEP_TRANSFORMER", None)

import core.runner as runner  # noqa: E402

# docs/RESIDENCY.md の収支表の行を、そのまま構成として表現したもの。
# `expect_gb` はドキュメントに載っている「必要」の値、`budget_gb` は同「実効予算」。
# `phase` はその行が見ている位相。
#
# 全ケースで `vae_resident=False` を明示しているのは、**この表が 48GB機の収支**だから。
# 48GB機は bnb-4bit の既定どおり VAE 対 (11GB) を CPU にパークする。走っている箱
# (統合メモリなら VAE 常駐が既定) の設定を混ぜ込むと、別の箱の表を再現できなくなる。
CASES = [
    {
        "name": "GPU0 48GB・デノイズ (TEは別GPU)",
        "phase": "denoise",
        "budget_gb": 49.81,
        "expect_gb": 40.60,
        "expect_ok": True,
        "config": dict(transformer_quant="int8", te_quant="bnb-4bit", te_prune=True, te_proj="", vae_resident=False),
        "te_external": True,
    },
    {
        "name": "GPU0 48GB・TE常駐のままデノイズ",
        "phase": "denoise",
        "budget_gb": 49.81,
        "expect_gb": 58.05,
        "expect_ok": False,
        "config": dict(transformer_quant="int8", te_quant="bnb-4bit", te_prune=True, te_proj="", vae_resident=False),
        "te_external": False,
    },
    {
        "name": "GPU0 48GB・transformer常駐のままデコード (fp32)",
        "phase": "decode",
        "budget_gb": 49.81,
        "expect_gb": 50.29,
        "expect_ok": False,
        "config": dict(
            transformer_quant="int8", video_vae_fp16=False, te_quant="bnb-4bit",
            te_prune=True, te_proj="", vae_resident=False,
        ),
        "te_external": True,
    },
    {
        "name": "同上 + video VAE fp16 (H3_KEEP_TRANSFORMER=1)",
        "phase": "decode",
        "budget_gb": 49.81,
        "expect_gb": 45.70,
        "expect_ok": True,
        "config": dict(
            transformer_quant="int8", video_vae_fp16=True, te_quant="bnb-4bit",
            te_prune=True, te_proj="", vae_resident=False,
        ),
        "te_external": True,
    },
    {
        "name": "単騎・投影TE同居で transformer も常駐 (plain + KEEP、fp16)",
        "phase": "denoise",
        "budget_gb": 49.81,
        # ドキュメントは 44.7 (デコード位相を 7.53GB とする新しい実測ベース)。
        # こちらは保守側の 6.6GB 活性化で 43.74 になる (差 ~1GB、判定は同じ)。
        "expect_gb": 44.7,
        "expect_ok": True,
        "config": dict(
            transformer_quant="int8", video_vae_fp16=True, te_quant="bnb-4bit",
            te_prune=True, te_proj="NicoLab28/ClipProj-MiniMax-H3", vae_resident=False,
        ),
        "te_external": False,
    },
]


def main() -> int:
    failures = []
    print(f"{'ケース':<52} {'必要':>8} {'資料':>8} {'予算':>8}  判定")
    print("-" * 92)
    for case in CASES:
        # TE 外部常駐は `_te_resident_gb()` が module 側の H3_TE_DEVICE を見るので差し替える。
        saved = runner.H3_TE_DEVICE
        runner.H3_TE_DEVICE = "cuda:1" if case["te_external"] else ""
        try:
            needs = runner._residency_requirements_gb(**case["config"])
        finally:
            runner.H3_TE_DEVICE = saved

        got_gb = needs[case["phase"]]
        ok = got_gb <= case["budget_gb"]
        verdict = "OK" if ok else "NG"
        mark = "  "
        if ok != case["expect_ok"]:
            mark = "<-判定不一致"
            failures.append(f"{case['name']}: 判定 {verdict}、資料は {'OK' if case['expect_ok'] else 'NG'}")
        elif abs(got_gb - case["expect_gb"]) > 0.5:
            mark = "<-必要量が資料と 0.5GB 超のずれ"
        print(
            f"{case['name']:<52} {got_gb:>8.2f} {case['expect_gb']:>8.2f} "
            f"{case['budget_gb']:>8.2f}  {verdict} {mark}"
        )

    print()
    budget = runner._effective_vram_budget_gb()
    if budget is None:
        print("この箱の実効予算: 測定不可 (旧来の直書き判定へフォールバックする)")
    else:
        import torch

        props = torch.cuda.get_device_properties(0)
        print(
            f"この箱の実効予算: {budget:.2f}GB "
            f"(total {props.total_memory / 1e9:.2f}GB, is_integrated={int(bool(getattr(props, 'is_integrated', 0)))})"
        )
        needs = {k: round(v, 2) for k, v in runner._residency_requirements_gb().items()}
        print(f"  現在の env での所要 (位相ごと): {needs}")

    if failures:
        print("\n判定不一致:")
        for f in failures:
            print("  -", f)
        return 1
    print("\n全ケースで判定が資料と一致した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
