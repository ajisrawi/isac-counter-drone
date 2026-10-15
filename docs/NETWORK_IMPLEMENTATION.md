# Implementing ISAC drone sensing in a 4G/5G network

This guide covers how to take the `isac` software from simulation to a live
operator network, step by step. It describes three deployment modes, where each
software block runs in the network, what the operator has to configure, and
how to validate performance.

---

## 1. The principle

```
          direct path (reference)
  eNB/gNB ───────────────────────────────►  sensing receiver
     │                                          ▲
     │  same downlink signal                    │ echo: delayed by (R_tx + R_rx - L)/c
     └────────────►  drone  ────────────────────┘        Doppler f_D = -(dR/dt)/λ
```

Each **Tx–Rx pair** measures two values per drone:

| measurement | meaning | resolution |
|---|---|---|
| bistatic excess range ρ = R_tx + R_rx − L | drone lies on an ellipsoid with foci Tx, Rx | c / occupied bandwidth (LTE 20 MHz ≈ 16.7 m, NR 100 MHz ≈ 3 m) |
| bistatic range rate ρ̇ | rate of change of the path length | λ / CPI (2.6 GHz, 100 ms ≈ 1.1 m/s) |

With **three or more pairs** the ellipsoids intersect in a 3-D fix. Range rates
give the velocity vector, and a Kalman tracker turns fixes into tracks.

Signal chain implemented in `isac/`:

```
IQ ─► sync (LTE: PSS, CFO, PCI) ─► OFDM demod ─► H = Y/X on known REs
   ─► common-phase correction ─► clutter removal (per-subcarrier mean, ECA)
   ─► range FFT ─► Doppler DFT ─► 2-D CA-CFAR ─► detections (ρ, ρ̇, SNR)
   ─► multistatic association + localisation ─► CV-Kalman tracks ─► C2 (JSON/UDP)
```

---

## 2. Three deployment modes

| | **A. Passive LTE** | **B. Network-integrated (4G or 5G)** | **C. Uplink sensing (5G)** |
|---|---|---|---|
| Operator cooperation | none (reception only) | full: Tx IQ / grid tap | gNB configures SRS |
| What is known | CRS pilots, regenerated from PCI | every transmitted RE | UE SRS sequences |
| Code path | `LtePassiveNode` | `TxTapNode`, `process_grids` | `process_grids` with SRS grid |
| Processing gain (10 MHz, 100 ms) | ~46 dB (CRS = 4.8 % of REs) | ~59 dB (all loaded REs) | depends on SRS config |
| Long-delay clutter (ECA) | no | yes | yes |
| Best use | fast pilot, proof of concept, covert sites | production national coverage | dense urban, extra geometry |

5G NR has no always-on reference signal like LTE CRS. Passive NR can only use
the SSB (one burst per 20 ms, 7–20 MHz wide), which gives ±25 Hz of unambiguous
Doppler, too little for drones. `NrSsbReference` exists for range-only work, but
**5G sensing in practice needs mode B or C**, i.e. operator integration.

---

## 3. Step-by-step rollout

### Step 0: Agreements and approvals (before any hardware)
1. **Customer and mandate.** Identify the national-security customer and the
   protected-site list (airports, borders, energy, stadiums, government).
2. **Regulator.** In Jordan this is the Telecommunications Regulatory Commission
   (TRC). Mode A only receives licensed signals; confirm that this is permitted
   for the sensing purpose. Modes B and C need the operator's licence holder on board.
3. **Operator agreement.** Agree site access, power and backhaul for receivers,
   and for mode B, access to baseband/fronthaul plus a non-interference guarantee.
4. **Privacy design.** Only public reference signals or the operator's own
   transmitted grid are used. No user data is decoded and no subscriber identities
   are touched. Write this down; regulators will ask.

### Step 1: Simulation and site planning (this repository)
```bash
python -m isac demo lte --plot results/lte.png      # passive 4G
python -m isac demo nr  --plot results/nr.png       # integrated 5G
```
- **Link budget.** Echo power = EIRP · G_rx · λ² · σ / ((4π)³ R_tx² R_rx²). With
  EIRP 60 dBm, a 10 dBi receiver and σ = 0.05 m², a drone 1 km from both ends is
  about −122 dBm at 2.6 GHz, and detectable after integration (see `sim.py`).
- **Geometry.** Plan **4 or more pairs** per protected point; 3 pairs work but
  leave little margin against ghosts. Baselines of 1–3 km work well. Receivers on
  different bearings give the best horizontal accuracy (low GDOP).
- **Altitude.** Masts sit at similar heights, so vertical accuracy is poor.
  The locator uses a soft altitude prior. For true 3-D, add an elevation-capable
  antenna array or EO/IR cueing.
- **Band choice.** Lower bands give longer range but coarser resolution:
  B20 800 MHz / B3 1800 MHz / B7 2600 MHz (LTE) or n78 3.5 GHz (NR, wide channels).

### Step 2: Lab validation
- Run an open-source base station (srsRAN Project / srsRAN 4G or OpenAirInterface)
  into an SDR, in a shielded room or on a licensed test channel.
- Mode A: check `acquire()` finds PCI, CFO and timing on the real waveform.
- Mode B: srsRAN/OAI can stream the transmitted baseband (ZMQ RF driver or a
  sample tap). Record it alongside the sensing receiver and process with `TxTapNode`.
- Use a corner reflector or a cooperative drone carrying a GNSS logger for ground truth.

### Step 3: Passive field pilot (mode A, about 3 months)
Per receiver site:

| item | example | why |
|---|---|---|
| SDR | USRP B210 / B205mini, bladeRF 2.0, LimeSDR | 2 phase-coherent channels on B210 (ref + surv) |
| clock | GPSDO (GNSS-disciplined 10 MHz + PPS) | <50 ppb so CPIs stay coherent; common time for fusion |
| antenna | 10–14 dBi panel, back toward the eNB | gain toward the area, >20 dB direct-path rejection |
| compute | 8-core mini-PC, 32 GB, or a Jetson Orin | the Python reference is offline; a GPU port runs in real time |
| backhaul | LTE modem / fibre to the fusion server | detections only, about 10 kB/s per pair |

Workflow:
```bash
# at each site, GNSS-triggered so all files start at the same second
python -m isac record --args "driver=uhd" --freq 2655e6 --rate 15.36e6 --seconds 30 --gain 45 --out rx1.cf32
# at the fusion server
python -m isac run examples/config_lte_passive.yaml -v > tracks.jsonl
```
Fly a cooperative drone on scripted patterns (radial, tangential, hover, altitude
steps) and compare tracks against its GNSS log.

### Step 4: Operator-integrated pilot (mode B, 4G or 5G)
Where each software block lives:

```
 ┌────────────── operator RAN ──────────────┐      ┌──── edge / MEC ────┐    ┌─ customer ─┐
 │ DU / baseband ── Tx grid or Tx IQ tap ───┼──►   │ sensing node(s):   │    │            │
 │ O-RAN 7.2x fronthaul ── U-plane mirror ──┼──►   │  TxTapNode /       │    │  C2 / COP  │
 │ sensing Rx (spare RU chain or SDR) ──────┼──►   │  process_grids     │──► │  (SAPIENT, │
 │ near-RT RIC xApp: sensing resource mgmt  │      │ FusionCenter:      │    │  ASTERIX,  │
 │ GNSS / PTP grandmaster (already present) │      │  locate + track    │    │  JSON)     │
 └──────────────────────────────────────────┘      └────────────────────┘    └────────────┘
```

**4a. Getting the transmitted signal (X)** – choose one:
1. **Fronthaul tap (O-RAN 7.2x).** The U-plane already carries the
   *frequency-domain* IQ of every PRB, usually block-floating-point compressed.
   Mirror the fronthaul switch port, decompress, map sections to PRBs using the
   C-plane, and feed `process_grids(Y, X, ...)` directly. No OFDM demod is needed for X.
2. **DU sample tap.** Copy the time-domain Tx samples (open-source DU, or a
   vendor sensing API) and feed `TxTapNode` with `tx=`.
3. **Reference antenna.** A high-gain antenna pointed at the cell (two-channel
   passive radar). Use it when no RAN access exists; the reference must be clean,
   at more than 30 dB SNR.

**4b. Getting the echo (Y).**
- **TDD (n78) caveat.** All synchronised cells transmit downlink at the same
  time, so a neighbour RU cannot receive the downlink echo during DL slots.
  Options: (i) a dedicated sensing receiver chain at the neighbour site
  (SDR or spare RU RX, isolated from its own Tx); (ii) schedule sensing bursts in
  guard periods or blanked symbols; (iii) uplink sensing (mode C).
- **FDD (LTE B3/B7/B20).** The downlink band is never used for reception at the
  site, so any spare receiver tuned to it works.
- **Time and frequency.** Use the operator's PTP/SyncE/GNSS. Required:
  time error < 100 ns between Tx tap and Rx (sets the absolute range bias, about 30 m),
  and frequency error < 50 ppb.

**4c. RAN configuration for 5G sensing (via the near-RT RIC xApp or OAM):**

| resource | sensing use | recommendation |
|---|---|---|
| SSB | sync, coarse range | default; too sparse for Doppler |
| TRS / CSI-RS | known, configurable, wideband | periodic bursts in every slot during sensing windows |
| PRS | positioning RS, comb 2–12, up to 272 PRB | best dedicated sensing RS: dense, full band, configurable muting |
| PDSCH + DMRS (full grid) | everything the DU sent | best gain; needs the 4a tap |
| beam sweeping | massive-MIMO beams toward protected sectors | the xApp steers SSB/CSI-RS beams over the protected zone |

**4d. Interfaces.** Detections and tracks go out as JSON (built in) over UDP/TCP.
For defence C2, wrap them in **SAPIENT** (BSI Flex 335, the common C-UAS sensor
interface) or **ASTERIX**. Keep the fusion server on the customer side of a
data diode if required.

### Step 5: Scale-out
- **Coverage.** Run the planning script over the operator's site database: for
  every grid point, count pairs with SNR above threshold, with a target of 4 or more.
- **Multi-operator.** Each operator's cells are additional illuminators; fusion is
  operator-agnostic because detections carry only Tx/Rx positions.
- **Sensor fusion.** Correlate tracks with RF detectors (Remote ID, control links),
  EO/IR cameras for confirmation, and the defeat layer.
- **Standards.** Follow 3GPP ISAC work (TR 22.837 requirements; Rel-19/20
  channel models and architecture studies) so the solution can move onto native
  gNB sensing functions when vendors ship them.

---

## 4. Validation KPIs

| KPI | target for a pilot | how |
|---|---|---|
| Pd (σ = 0.01–0.05 m² drone) | ≥ 0.9 inside the planned zone | cooperative drone, 100+ passes |
| False tracks | < 1 per hour per km² | 24 h runs without drones |
| Horizontal accuracy | < 20 m (1σ) | track vs drone GNSS |
| Velocity accuracy | < 1 m/s | track vs drone GNSS |
| Latency | < 1 s detection to C2 | timestamps |
| RAN impact (mode B) | zero KPI degradation | operator counters before/after |

What the simulator shows (realistic link budget, `python -m isac demo ...`):

| scenario | per-pair accuracy | track accuracy |
|---|---|---|
| passive LTE 10 MHz, CRS, CPI 100 ms, 4 Rx | ≈ 1–5 m range, ≈ 0.1–0.3 m/s rate | ≈ 2 m horizontal |
| integrated NR 20 MHz, CPI 40 ms, 3 Rx | ≈ 0.3–1 m range, ≈ 0.05 m/s rate | ≈ 1 m horizontal |

These are simulation figures (free space, ideal clocks, no birds or wind-blown
clutter). Expect several times worse in the field, which is what Step 3 is for.

---

## 5. Known limitations and mitigations

| limitation | effect | mitigation |
|---|---|---|
| Hovering drone (≈ 0 Doppler) | cancelled together with static clutter | longer CPI, slower clutter filter, rotor micro-Doppler (future), EO cue |
| Clutter beyond the cyclic prefix | ISI raises the noise floor (+25 dB seen in sim) | ECA in modes B and C (`clutter.eca`); site antennas to avoid big reflectors |
| Birds | false alarms at drone-like speeds | micro-Doppler / track-behaviour classifier (planned) |
| Low CRS density (mode A) | ~13 dB less gain than mode B | 20 MHz carriers, both CRS ports (`ports=(0, 1)`), 100–200 ms CPI |
| SDR clock drift | direct-path delay drift → residual clutter | GPSDO; per-CPI direct-path tracking |
| Altitude | poor vertical accuracy | elevation array, altitude prior, EO/IR |
| Compute | Python reference is offline | GPU port (CuPy drop-in for numpy, or NVIDIA Aerial / cuPHY on the DU) |

---

## 6. Software module map

| module | role | runs at |
|---|---|---|
| `numerology.py`, `ofdm.py` | LTE/NR frame structure, CP-OFDM | sensing node |
| `lte.py`, `nr.py`, `sequences.py` | CRS, PSS, SSB, Gold sequences | sensing node |
| `sync.py` | passive acquisition (PSS, CFO, PCI) | sensing node (mode A) |
| `reference.py` | which REs are known (CRS / SSB / full grid) | sensing node |
| `clutter.py` | ECA time-domain cancellation | sensing node (modes B, C) |
| `processing.py`, `cfar.py` | range-Doppler, CFAR, detections | sensing node / DU dApp |
| `localize.py`, `tracker.py` | multistatic fix, Kalman tracking | MEC fusion server |
| `pipeline.py` | node and fusion orchestration | both |
| `sources.py`, `cli.py` | SDR recording, file replay, JSON/UDP output | both |
| `sim.py`, `demo.py` | physics simulator, end-to-end demos | lab |
