PROMPT MESTRE (cole no Codex/GPT e mande gerar o código)

Você é um engenheiro sênior de MQL5 e DSP. Gere um indicador MetaTrader 5 (MQL5) chamado:
DominantCycles_STFT_Kalman_Hilbert_SG.mq5

Requisitos INEGOCIÁVEIS:
- ZERO parâmetros “fake”: todo `input` precisa ser usado no cálculo e alterar a saída de forma mensurável.
- Oscilador LIVRE: sem normalização para [-1,1], sem saturação, sem teto/chão, sem clamp de amplitude.
- Robusto em dojis e barras flat: nenhuma divisão por (range, vol, etc) que gere spikes; use eps apenas onde inevitável.
- Causal por padrão (sem repaint). Se oferecer modo zero‑phase/simétrico, ele deve ser explicitamente “REPAINT/NON‑CAUSAL” e desligado por padrão.
- O indicador deve fornecer continuamente entre 3 e 7 ciclos dominantes (K selecionável) a cada barra, com STFT + Kalman + Hilbert + Savitzky‑Golay, com unwrap de fase e cálculo/exposição de group delay.
- Deve ser utilizável em EA via iCustom: buffers para saída principal, ciclos individuais, atributos por ciclo, direção e flip.

ALGORITMO (implemente exatamente, não descreva apenas):

1) Série de entrada
Input enum PriceSource {CLOSE, HLC3, OHLC4}. Use preço em unidades reais (sem normalização por range).

2) Detrend sem distorção desnecessária
Input enum DetrendMode {KALMAN_LLT, SG_ENDPOINT}.
- KALMAN_LLT: Filtro de Kalman local linear trend com estado [level, slope], medição price. Saída trend=level.
- SG_ENDPOINT: Savitzky–Golay por regressão polinomial em janela trailing W (input), ordem p (input), avaliado no último ponto (endpoint), trend = y_hat(t). Causal e sem atraso extra.
Residual: resid = price - trend.
Exponha trend e resid em buffers.

3) STFT (causal trailing window) em resid
A cada barra fechada (default), compute espectro nos últimos N pontos de resid:
- Janela Hann (obrigatório). (Se oferecer Blackman, use de fato e com input).
- Calcule apenas bins cujo período esteja dentro [MinPeriodBars..MaxPeriodBars]. Garanta bins suficientes.
- Use Goertzel por bin (preferível) ou FFT radix-2 (aceitável). Precisa retornar coeficiente complexo (Re/Im) para amplitude e fase.
- Selecione Kcand >= 2*K picos por potência, com critério de máximo local.
- Refine frequência de cada pico com interpolação parabólica em log‑potência (ou Jacobsen/Quinn). Depois recompute o coeficiente complexo na frequência refinada para obter amp_raw e phase_raw.
- Calcule SNR por pico: Ppeak / (mean Prest + eps). Exponha SNR (linear e dB).

4) Rastreamento multi-ciclo (3..7 ciclos) com associação de dados
Mantenha K tracks persistentes ao longo do tempo (IDs estáveis):
Para cada track k:
- Kalman em omega: estado [omega, omega_dot].
- Kalman em amplitude: estado [A, A_dot].
- Fase: mantenha phase_unwrapped_k com previsão phase_pred = phase_prev + omega_true_prev*Δt. Faça unwrap do measurement phase_raw/hilbert ao redor da previsão (evitar saltos ±π).
A cada barra:
a) Prediga omega/amp/fase de cada track.
b) Associe picos medidos aos tracks por custo (|omega_meas-omega_pred|/sigma + penalização por SNR baixo). Use matching determinístico (greedy é ok para K<=7).
c) Atualize os tracks associados com medições (omega_meas, A_meas, phase_meas), com R adaptativo por SNR.
d) Tracks não associados: reduza confiança; após Nmiss, permitir substituição por pico novo.
e) Se faltarem tracks bons, crie novos a partir dos melhores picos não usados.
Ordene tracks por potência/confiança, mas mantenha IDs internos para evitar “troca” de ciclo que causa descontinuidade.

5) Hilbert (analítico) por ciclo (fusão real)
Implemente FIR Hilbert transform (comprimento L ímpar, input), windowed (Hamming ou similar).
- Group delay do Hilbert: (L-1)/2. Exponha em buffer/variável.
Use Hilbert de duas formas (faça de verdade):
A) Aplicar no sinal narrowband do ciclo:
   - Gere x_k(t) usando reconstrução por cos(phase) OU preferencialmente bandpass ao redor de omega_true (resonator/IIR adaptativo).
   - Compute quadratura via Hilbert: y_k = H{x_k}. Analytic: a_k = x_k + j y_k.
   - Extraia amp_h e phase_h.
B) Fusão de medições:
   - Combine (phase_raw_STFT, amp_raw_STFT) com (phase_h, amp_h) em atualização Kalman, com pesos por SNR/confiança.
O objetivo é reduzir tremedeira de fase e amplitude.

6) Savitzky–Golay smoothing (atributos) sem “fake”
Use SG para suavizar omega_true e amp_true (e opcionalmente derivada da fase), mas:
- default: SG endpoint (causal).
- opcional: SG symmetric (REPAINT) com group delay = (W-1)/2 (se W ímpar). Exponha group delay e aplique compensação somente no modo REPAINT.

7) Reconstrução dos ciclos e saída principal
Para cada track k:
cycle_k[t] = A_true_k[t] * cos(phase_true_k[t]).
Sem normalização/clamp.
Modo de plot (input PlotMode):
- SUM_TOP_K (default): soma dos K ciclos mais confiáveis
- SUM_TOP2
- SINGLE_INDEX (input CycleIndex)
- MASK (input bitmask 1..7)
Implemente todos e use os inputs de verdade.
Saída principal OutputMain = conforme modo.

8) Direção, flip e coloração para EA
Buffers:
- Dir[t] = +1 se OutputMain[t] > OutputMain[t-1], -1 se menor, 0 se igual (barra fechada).
- Flip[t] = +1 quando Dir muda de -1 para +1; -1 quando muda de +1 para -1; 0 caso contrário.
Color plot: verde quando Dir=+1, vermelho quando Dir=-1. Não dependa de tick; deve funcionar em barras fechadas.

9) Forecast real (não “igual ao atual”)
Input ForecastHorizonBars H (>=1).
Propague fase: phase(t+H)=phase(t)+omega_true*H.
ForecastMain[t] = soma dos ciclos usando amp_true (com decay opcional realmente implementado).
Exponha ForecastMain em buffer (numérico).
Opcional: plot deslocado (shift) para a direita se PlotForecastShift=true (input usado).

10) Performance e robustez
- Heavy spectral calc apenas em barra nova quando CalcOnEveryTick=false (default).
- Precompute: window weights, sin/cos por bin, coeficientes SG, coeficientes Hilbert.
- Use epsilons para evitar NaN/INF sem criar spikes.
- Garanta que o indicador compile sem warnings.

DELIVERÁVEIS no final da sua resposta:
1) O código COMPLETO do arquivo `DominantCycles_STFT_Kalman_Hilbert_SG.mq5` (sem omissões), pronto para compilar.
2) Um `README.md` com:
   - explicação dos modos,
   - mapa exato de buffers e índices para iCustom/CopyBuffer,
   - presets recomendados para EURUSD M15 e para índices/cripto.
3) (Opcional mas recomendado) `ExampleEA_iCustom.mq5` minimalista que lê:
   - OutputMain, Dir, Flip, SNR médio dos ciclos, PeriodTrue do ciclo #1,
   e abre/fecha posição apenas em barra fechada, com filtro por SNR.

CONDIÇÃO FINAL: se você não implementar algum item acima, declare explicitamente qual item falhou e por quê, e corrija antes de finalizar. Não invente parâmetro que não use.



------------------------------------------------------------------------------------------------------------------







SUA MENSAGEM AO COMEÇAR A IMPLEMENTACAO DO INDICADOR:




   // trade on flips
    if(flip>0.0 && dir>0.0)
diff --git a//mnt/c/mql/indicators/stfft-kalman/README.md b//mnt/c/mql/indicators/stfft-kalman/README.md
index 9209429d6f86790268a85df33130f4daa72cd61c..4eae13b42f1f1c2f8fadb729a76a5a01a57b7667
--- a//mnt/c/mql/indicators/stfft-kalman/README.md
+++ b//mnt/c/mql/indicators/stfft-kalman/README.md
@@ -84,3 +84,7 @@
 - Default mode is causal (no repaint). If you enable symmetric SG, it will repaint.
 - Use `InpPlotMode=PLOT_MASK` with `InpCycleMask` bits 1..7 to select specific cycles.
 - For EA usage, prefer reading shift=1 (closed bar) unless `InpCalcOnTick=true`.
+
+## Example EA
+- `ExampleEA_iCustom.mq5` reads OutputMain, Dir, Flip, average SNR, and Cycle #1 period.
+- Ensure `InpKCycles` in the EA matches the indicator setting.

tokens used
199,277
Implemented the full dominant‑cycle oscillator pipeline with STFT peak tracking, Kalman attribute filters, Hilbert refinement, SG smoothing, and the full EA‑ready buffer map, then documented usage and added a minimal EA example.

Details:
- `DominantCycles_STFT_Kalman_Hilbert_SG.mq5`: full indicator with causal default, explicit REPAINT mode, STFT+Goertzel peak refinement, multi‑track Kalman association, Hilbert analytic fusion weighted by SNR/confidence, SG endpoint/symmetric smoothing, forecast, direction/flip, and fixed buffer index map.
- `README.md`: algorithm description, causal vs REPAINT behavior, full buffer index map, presets, and EA notes.
- `ExampleEA_iCustom.mq5`: minimal EA reading OutputMain/Dir/Flip, average SNR gate, and cycle‑1 period on closed bars.

Tests not run (MetaEditor compile not available here).

Next steps:
1. Compile `DominantCycles_STFT_Kalman_Hilbert_SG.mq5` and `ExampleEA_iCustom.mq5` in MetaEditor.
2. Load the indicator on a chart and validate parameter sensitivity (Min/MaxPeriod, K, Hilbert on/off, SG mode).
3. Run the EA in the Strategy Tester on closed bars (shift=1) to verify signal behavior.







==================================================================================================================




EU TENTEI COMPILAR VIERAM ESSES ERROS:



DominantCycles_STFT_Kalman_Hilbert_SG.mq5			
idenfitier 'PRICE_CLOSE' already used	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	13	4
   built-in 'ENUM_APPLIED_PRICE::PRICE_CLOSE'			0
implicit conversion from 'enum ENUM_APPLIED_PRICE' to 'enum PriceSource'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	45	34
   'PriceSource::PRICE_HLC3' will be used instead of 'ENUM_APPLIED_PRICE::PRICE_CLOSE'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	14	4
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	129	27
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	130	27
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	131	28
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	132	25
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	133	30
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	134	28
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	135	25
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	136	27
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	137	26
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	140	30
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	141	28
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	178	28
'[' - invalid index value	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	197	32
'g_cycle' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	676	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	676	31
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	676	7
'g_omega' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	677	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	677	31
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	677	7
'g_period' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	678	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	678	32
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	678	7
'g_amp' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	679	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	679	29
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	679	7
'g_phaseUnw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	680	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	680	34
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	680	7
'g_phaseW' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	681	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	681	32
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	681	7
'g_snr' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	682	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	682	29
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	682	7
'g_snrDb' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	683	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	683	31
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	683	7
'g_conf' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	684	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	684	30
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	684	7
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	685	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	685	34
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	685	7
'g_ampRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	686	24
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	686	32
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	686	7
'g_cycle' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	689	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	689	36
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	689	7
'g_omega' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	690	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	690	36
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	690	7
'g_period' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	691	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	691	37
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	691	7
'g_amp' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	692	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	692	34
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	692	7
'g_phaseUnw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	693	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	693	39
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	693	7
'g_phaseW' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	694	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	694	37
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	694	7
'g_snr' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	695	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	695	34
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	695	7
'g_snrDb' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	696	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	696	36
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	696	7
'g_conf' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	697	29
cannot convert parameter 'double' to 'double&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	697	35
   built-in: bool SetIndexBuffer(int,double&[],ENUM_INDEXBUFFER_TYPE)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	697	7
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	777	22
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	777	32
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	777	10
'g_ampRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	778	22
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	778	30
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	778	10
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	779	27
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	779	37
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	779	10
'g_ampRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	780	27
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	780	35
   built-in: bool ArraySetAsSeries(const T&[...],bool)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	780	10
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	781	26
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	781	36
   built-in: int ArrayInitialize(T&[...],T)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	781	10
'g_ampRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	782	26
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	782	34
   built-in: int ArrayInitialize(T&[...],T)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	782	10
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	790	27
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	790	37
   built-in: int ArraySize(const T&[...])	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	790	17
'g_omegaRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	793	25
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	793	35
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	793	13
'g_ampRaw' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	794	25
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	794	33
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	794	13
'g_prevBpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	834	22
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	834	34
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	834	10
'g_bpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	865	22
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	865	30
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	865	10
'g_bpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1006	23
cannot convert parameter 'double' to 'const void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1006	31
   built-in: int ArraySize(const T&[...])	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1006	13
'g_bpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1007	25
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1007	33
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	1007	13
'g_bpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	486	22
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	486	30
   built-in: int ArrayResize(T&[...],int,int)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	486	10
'g_bpHist' - invalid array access	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	487	26
cannot convert parameter 'double' to 'void&[]'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	487	34
   built-in: int ArrayInitialize(T&[...],T)	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	487	10
implicit conversion from 'enum ENUM_APPLIED_PRICE' to 'enum PriceSource'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	634	23
   'PriceSource::PRICE_HLC3' will be used instead of 'ENUM_APPLIED_PRICE::PRICE_CLOSE'	DominantCycles_STFT_Kalman_Hilbert_SG.mq5	14	4
84 errors, 2 warnings		84	2







Voce deve ir fazendo as correcoes e compilando até que nao haja mais erros. Para isto voce usa o comando "cmdmt", implementacao local, que é bem simples e permite voce compilar e ver depuracao em tempo real, com traceback, bem como voce pode adicionar o indicador no gráfico e ver seus logs para saber se está tudo ok. 


Tudo isto com o "cmdmt".
