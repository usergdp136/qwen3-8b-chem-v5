# qwen3-8b-chem-v5

## Italiano

**Che cos'è.** Qwen3-8B (8B, non 14B) con l'adattatore QLoRA «chem v5» (rango 16, α = 32, circa 43,6 M parametri addestrabili) fuso nei pesi e quantizzato in GGUF Q4_K_M (circa 5 GB) per Ollama. L'adattatore è stato addestrato per un'epoca su 11.839 esempi di chimica con licenze permissive, su una sola GPU da portatile da 12 GB.

**Come usarlo.** Il GGUF contiene già il modello base fuso con l'adattatore: non serve scaricare Qwen3-8B a parte né usare PEFT.

```
ollama pull ollamagdp/qwen3-8b-chem-v5
ollama run ollamagdp/qwen3-8b-chem-v5
```

Serve un Ollama aggiornato e circa 5 GB di disco. Sono consigliati almeno 8 GB di RAM o VRAM; con meno memoria funziona, più lentamente, in parte su CPU. Il modello può ragionare in modo visibile («thinking»): via API usare `"think": false` per allinearsi alla valutazione.

**Valutazione** (test sigillato e sviluppo, base contro adattatore, 4 bit NF4, zero-shot, non-thinking, punteggio sulle probabilità delle lettere, bootstrap appaiato 10.000 campioni):

| Split | Benchmark | n | Base | + Adattatore | Differenza [IC 95 %] |
|---|---|---:|---:|---:|---|
| Test sigillato | Complessivo | 2529 | 47,61 % | 52,63 % | **+5,02 [+3,24; +6,76]** superato |
| Test sigillato | MMLU chimica | 233 | 63,95 % | 67,81 % | +3,86 [−0,43; +8,15] |
| Test sigillato | ChemBench MCQ | 1503 | 54,82 % | 58,22 % | +3,39 [+1,13; +5,72] |
| Test sigillato | MMLU-Pro chimica | 793 | 29,13 % | 37,58 % | +8,45 [+5,17; +11,73] |
| Sviluppo | Complessivo | 1083 | 50,69 % | 51,89 % | +1,20 [−1,29; +3,79] non superato |
| Sviluppo | MMLU chimica | 100 | 64,00 % | 63,00 % | −1,00 [−6,00; +4,00] |
| Sviluppo | ChemBench MCQ | 644 | 60,09 % | 58,70 % | −1,40 [−4,97; +2,02] |
| Sviluppo | MMLU-Pro chimica | 339 | 28,91 % | 35,69 % | +6,78 [+2,36; +11,21] |

Test sigillato (17 novembre 2026, n = 2.529): 328 domande guadagnate, 201 perse; netmon ha registrato 0 connessioni. `mmlupro_chem` è stato rilanciato da solo dopo un segfault nella run 1.

**Attenzione:** questo GGUF (adattatore fuso + 4 bit) non è l'oggetto valutato nell'articolo; i numeri sopra si riferiscono all'adattatore in 4 bit NF4 sul modello base, non a questa quantizzazione.

**Per caricare l'adattatore grezzo con PEFT / Transformers** (non serve se usi Ollama): scarica prima **Qwen/Qwen3-8B** (8B, non 14B), poi `PeftModel.from_pretrained(..., "odelpa/qwen3-8b-chem-v5", subfolder="adapter")`, con `enable_thinking=False`. Si può usare la stessa quantizzazione 4 bit NF4 della valutazione, oppure fondere i pesi con `merge_and_unload()`. Esempio completo: [scheda su Hugging Face](https://huggingface.co/odelpa/qwen3-8b-chem-v5) e `examples/load_adapter.py` sul [GitHub](https://github.com/usergdp136/qwen3-8b-chem-v5).

**Link.** Articolo (DOI): [10.5281/zenodo.24326182](https://doi.org/10.5281/zenodo.24326182) · Adattatore: [Hugging Face](https://huggingface.co/odelpa/qwen3-8b-chem-v5) · Codice e scheda: [GitHub](https://github.com/usergdp136/qwen3-8b-chem-v5)

**Licenza.** Apache-2.0. Modello base Qwen3-8B © Alibaba Cloud / Qwen team, Apache-2.0. Una parte dei dati di addestramento deriva da Wikipedia (CC BY-SA 4.0).

**Autore.** Giovanni Del Papa, ricercatore indipendente, Italia · giovanni.delpapa@live.it

**Limiti.** Scarto tra sviluppo (gate non superato) e test sigillato (gate superato); crash e rilancio di un benchmark; valutazione in 4 bit, non su questo GGUF; un solo run e un solo seed; nessuna valutazione di sicurezza; nessuna garanzia chimica, medica, legale o professionale.

---

## English

**What it is.** Qwen3-8B (8B, not 14B) with the «chem v5» QLoRA adapter (rank 16, α = 32, about 43.6 M trainable parameters) merged into the weights and quantized to GGUF Q4_K_M (about 5 GB) for Ollama. The adapter was trained for one epoch on 11,839 permissively licensed chemistry examples, on a single 12 GB laptop GPU.

**How to run it.** The GGUF already contains the merged base, so nothing else needs downloading: no separate Qwen3-8B and no PEFT.

```
ollama pull ollamagdp/qwen3-8b-chem-v5
ollama run ollamagdp/qwen3-8b-chem-v5
```

You need a recent Ollama and about 5 GB of disk. At least 8 GB of RAM or VRAM is recommended; with less it still runs, more slowly, partly on CPU. The model can reason visibly («thinking»): through the API use `"think": false` to match the evaluation setting.

**Evaluation** (sealed test and development, base vs adapter, 4-bit NF4, zero-shot, non-thinking, letter-probability scoring, paired bootstrap 10,000 resamples):

| Split | Benchmark | n | Base | + Adapter | Difference [95 % CI] |
|---|---|---:|---:|---:|---|
| Sealed test | Pooled | 2529 | 47.61 % | 52.63 % | **+5.02 [+3.24; +6.76]** PASS |
| Sealed test | MMLU chemistry | 233 | 63.95 % | 67.81 % | +3.86 [−0.43; +8.15] |
| Sealed test | ChemBench MCQ | 1503 | 54.82 % | 58.22 % | +3.39 [+1.13; +5.72] |
| Sealed test | MMLU-Pro chemistry | 793 | 29.13 % | 37.58 % | +8.45 [+5.17; +11.73] |
| Development | Pooled | 1083 | 50.69 % | 51.89 % | +1.20 [−1.29; +3.79] FAIL |
| Development | MMLU chemistry | 100 | 64.00 % | 63.00 % | −1.00 [−6.00; +4.00] |
| Development | ChemBench MCQ | 644 | 60.09 % | 58.70 % | −1.40 [−4.97; +2.02] |
| Development | MMLU-Pro chemistry | 339 | 28.91 % | 35.69 % | +6.78 [+2.36; +11.21] |

Sealed test (17 November 2026, n = 2,529): 328 items gained, 201 lost; netmon recorded 0 connections. `mmlupro_chem` was re-run alone after a segfault in run 1.

**Note:** this GGUF (merged adapter + 4-bit) is not the object evaluated in the paper; the figures above refer to the 4-bit NF4 adapter on the base model, not to this quantization.

**To load the raw adapter with PEFT / Transformers** (not needed if you use Ollama): download **Qwen/Qwen3-8B** first (8B, not 14B), then `PeftModel.from_pretrained(..., "odelpa/qwen3-8b-chem-v5", subfolder="adapter")`, with `enable_thinking=False`. You can use the same 4-bit NF4 option as the evaluation, or merge the weights with `merge_and_unload()`. Full snippet: [Hugging Face model card](https://huggingface.co/odelpa/qwen3-8b-chem-v5) and `examples/load_adapter.py` on [GitHub](https://github.com/usergdp136/qwen3-8b-chem-v5).

**Links.** Paper (DOI): [10.5281/zenodo.24326182](https://doi.org/10.5281/zenodo.24326182) · Adapter: [Hugging Face](https://huggingface.co/odelpa/qwen3-8b-chem-v5) · Card and code: [GitHub](https://github.com/usergdp136/qwen3-8b-chem-v5)

**License.** Apache-2.0. Base model Qwen3-8B © Alibaba Cloud / Qwen team, Apache-2.0. Some training data is derived from Wikipedia (CC BY-SA 4.0).

**Author.** Giovanni Del Papa, Independent researcher, Italy · giovanni.delpapa@live.it

**Limits.** Dev/test gap (development gate failed, sealed-test gate passed); crash and re-run of one benchmark; evaluation in 4-bit, not on this GGUF; one run, one seed; no safety evaluation; no chemical, medical, legal or other professional guarantee.
