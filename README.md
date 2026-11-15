# CrackLens

**Crack segmentation for concrete and building surfaces, with scale-free geometry statistics and an evaluation built to show where it fails.**

CrackLens takes a photo of a concrete or building surface and returns:

1. a **crack mask** (which pixels are crack), and
2. a set of **geometry statistics** computed from that mask (how wide, how long, how branched, how much of the image), expressed in pixels and ratios, never in millimetres unless the caller supplies a scale.

It is served as a Dockerized API with a tested, reproducible evaluation pipeline.

> **What CrackLens is not.** It is a segmentation system with unusually rigorous evaluation. It is **not a measurement tool**, **not a structural-safety assessment**, and must not be used to make safety decisions. A single photo gives pixels, not millimetres (see [Why no millimetres by default](#why-no-millimetres-by-default)).

---

## Contents

1. [Aim](#aim)
2. [Why this project exists](#why-this-project-exists)
3. [Scope](#scope)
4. [Glossary](#glossary)
5. [System overview](#system-overview)
6. [Data](#data)
7. [Evaluation protocol](#evaluation-protocol)
8. [Metrics](#metrics)
9. [Models](#models)
10. [Inference on large images](#inference-on-large-images)
11. [Geometry module](#geometry-module)
12. [Experiments](#experiments)
13. [Serving](#serving)
14. [Testing](#testing)
15. [Reproducibility and experiment discipline](#reproducibility-and-experiment-discipline)
16. [Repository layout](#repository-layout)
17. [Results](#results)
18. [Limitations](#limitations)
19. [Questions this project is built to answer](#questions-this-project-is-built-to-answer)
20. [Definition of done](#definition-of-done)
21. [Licenses, data terms and citations](#licenses-data-terms-and-citations)

---

## Aim

Build a crack-segmentation system and **measure honestly how well it works**, especially on photos it was never trained on.

Concretely, the project aims to:

- **Quantify the cross-dataset gap:** how much worse the model is on a different dataset, different surface type, and different camera than it was trained on. This gap is the headline result.
- **Test one controlled design choice:** whether adding crack-free "hard negative" images (stains, peeling paint, algae, spalling, plain wall) to training reduces false alarms, and whether that holds up on held-out data.
- **Validate the geometry module against ground truth:** how accurately widths computed from a mask match widths computed from the human-drawn mask, binned by how thin the crack is.
- **Show what happens to thin cracks when images are shrunk.**
- **Deliver a deployable system:** exported model, measured CPU latency, API, container, automated tests.
- **Document failures,** not hide them: a failure gallery and an honest limitations section.

Success is measured by evidence in this repository (tables, tests, runnable code), not by a single headline accuracy number.

## Why this project exists

Public crack datasets are small, come from a few sources, and are dominated by pavement. Models that score well on their own dataset often fail on a different one, and many published comparisons mix datasets that share duplicate images, which inflates results. A system that is only evaluated on images that look like its training data tells you little about how it behaves on a new wall.

CrackLens treats the evaluation as the product. The held-out set is chosen and locked before training, scores are reported as per-image distributions, geometry is validated against ground truth, and failure modes are catalogued.

## Scope

### In scope

- Binary crack segmentation (crack / not crack) on photos of concrete and building surfaces.
- Training and evaluation on public datasets with pixel-level masks.
- Hard-negative training using image-level-labelled datasets (all-zero masks for crack-free defect images).
- Scale-free geometry statistics computed from the mask.
- Optional unit conversion when the caller supplies `px_per_mm`.
- Metrics implemented from scratch with unit tests.
- A classical (no learning) baseline and a U-Net baseline with a pretrained encoder.
- Tiled inference at native resolution for large photos.
- Error analysis, failure gallery, resolution experiment.
- ONNX export, CPU latency measurement, FastAPI service, Dockerfile, CI that runs the tests.

### Out of scope (deliberately)

| Excluded | Reason |
|---|---|
| Millimetre width or length without a supplied scale | A single image has no scale. See below. |
| Severity grading | Inspection rubrics are keyed to physical width in mm, which cannot be derived honestly without a scale. |
| Multi-class defects (spalling, algae, stains, honeycombing as prediction targets) | This is a single-target crack system. Other defects appear only as crack-free negatives. |
| UAV / drone imagery | Different scale and capture conditions. |
| Change tracking between visits | Needs registration and repeat captures. |
| SegFormer and other model-family comparisons | A pretrained-encoder U-Net is the baseline; architecture shopping is not the point. |
| Uncertainty maps, test-time augmentation, calibration | Not needed to answer the project's questions. |
| Drift monitoring, Prometheus/Grafana, Kubernetes | Production operations are out of scope. |
| Video | Single images only. |
| LiDAR or depth-based scale recovery | Deferred. It would give millimetres per pixel for roughly fronto-parallel surfaces, but it is a separate project. |
| Job queue and cloud deployment | The API is synchronous; this is documented as a limitation. |

### Why no millimetres by default

Pixels convert to millimetres only if you know how many millimetres one pixel covers. For a single photo that requires either the camera-to-surface distance plus the camera's focal length and sensor details, or a reference object of known size in the photo. CrackLens has neither by default.

So the API accepts an **optional** `px_per_mm`. If supplied, width and length are also returned in millimetres. If not, everything is reported in pixels or unitless ratios, and the response says so explicitly.

## Glossary

| Term | Meaning |
|---|---|
| **Mask** | A black-and-white image the same size as the photo. Crack pixels are 1, everything else is 0. |
| **Ground-truth mask** | The mask a human drew. Treated as the reference answer, though it is itself annotator-dependent. |
| **Segmentation** | Deciding, for every pixel, whether it belongs to a crack. |
| **IoU (overlap score)** | Pixels both masks mark as crack, divided by pixels either mask marks. 1 is perfect, 0 is no overlap. |
| **Dice** | 2 × pixels both mark, divided by (crack pixels in prediction + crack pixels in truth). |
| **Thin-line score (centerline IoU)** | Both masks are thinned to 1-pixel lines; a predicted line counts as correct if it lies within a tolerance (4 px) of a true line. Fairer than IoU for very thin cracks. |
| **Skeleton** | A mask thinned to a 1-pixel-wide centerline. |
| **Distance transform** | For each crack pixel, the distance to the nearest non-crack pixel. Sampled along the skeleton, twice this distance approximates local crack width. |
| **Hard negative** | A photo with no crack but with something crack-like (stain, peeling, algae, spalling). Its mask is all zeros. Used to teach the model what is not a crack. |
| **Cut-off (threshold)** | The probability above which a pixel is called crack. Chosen on validation data only. |
| **Training / validation / test** | Training: what the model learns from. Validation: used to make choices (cut-off, when to stop). Test: used only to report scores. |
| **In-domain test** | Test images from the same datasets as training (their own test splits). |
| **Held-out set ("exam set")** | A dataset locked before training and never used for any choice. Measures cross-dataset behaviour. |
| **Cross-dataset gap** | The score difference between in-domain test and the held-out set. |
| **Tile / tiled inference** | Cutting a large photo into overlapping pieces, predicting each, and merging the predictions. |
| **False-alarm rate** | On crack-free images, how often the model predicts crack. |
| **ONNX** | A portable model file format that runs without PyTorch. |
| **p50 / p95 latency** | Median and 95th-percentile time to process one tile. |
| **`px_per_mm`** | Optional caller-supplied scale: how many pixels span one millimetre in this photo. |

## System overview

```
                 +-----------------------------------------------+
 photo  ───────► |  tile (overlapping pieces at native resolution)|
 (+ optional     +-----------------------+-----------------------+
  px_per_mm)                             │
                                         ▼
                             +-----------------------+
                             |  U-Net (ONNX runtime) |
                             +-----------+-----------+
                                         │ per-tile probabilities
                                         ▼
                             +-----------------------+
                             | stitch (average overlaps) |
                             +-----------+-----------+
                                         │ probability map
                                         ▼
                             +-----------------------+
                             | cut-off (from validation) |
                             +-----------+-----------+
                                         │ binary mask
                                         ▼
                             +-----------------------+
                             |   geometry module     |
                             | skeleton, distance    |
                             | transform, counts     |
                             +-----------+-----------+
                                         │
                                         ▼
                      mask PNG  +  JSON statistics (px / ratios,
                                   mm only if px_per_mm supplied)
```

Evaluation shares the same tiling, stitching, cut-off and geometry code as serving, so what is measured is what is shipped.

## Data

### Dataset roles

| Dataset | Masks? | Role |
|---|---|---|
| **Crack500** | Yes | Training, validation, in-domain test |
| **DeepCrack** (537 images) | Yes | Training, validation, in-domain test |
| **Özgenel Concrete Crack Segmentation Dataset** (458 photos) | Yes | **Held-out set.** Locked before training. Never used for any choice. |
| **BD3** (IISc) | Image-level labels only | Source of **hard negatives for training** |
| **ConViD** | Image-level labels only | **Test only:** false-alarm set plus a small hand-annotated crack set |

Pooled training uses Crack500 and DeepCrack together, plus hard negatives when the ablation calls for them. In-domain test scores are reported **separately per dataset**.

### Dataset notes and verification status

Status key: **Verified** means checked against the dataset's own page or repository. **To verify** means not yet confirmed and must be confirmed (and recorded in `DATA.md`) before use.

**Özgenel (held-out set)**
- Verified: 458 high-resolution photos with black-and-white crack maps, captured on buildings at a single university campus; license CC BY 4.0.
- To verify: mask file format and polarity (crack = white or black), one-to-one pairing of photos and masks, exact image size of this download (the sister classification dataset states 4032×3024 for the source photos).
- Caveat: all photos come from **one campus**. The held-out result describes generalisation to a different campus's buildings, not to buildings in general.

**Crack500**
- Verified (via third-party descriptions and the original authors' repository): about 500 phone photos, distributed as small crops (commonly 1,896 train / 348 validation / 1,124 test); described as pavement cracks only. The authors' repository asks for citation and, in the README, states no license.
- To verify: that the train/val/test split is **by original photo** and not by crop. File names appear to encode the original photo, and the check is to group file names by original photo and confirm no photo appears in more than one split. If any does, the data is re-split by photo.
- To verify: image sizes, and that the content is pavement (visual check of a sample).
- Consequence: Crack500 is a **domain mismatch** with building concrete.

**DeepCrack**
- Verified: 537 images (official split 300 train / 237 test); the authors restrict use of their images and annotations to non-commercial research and education.
- Naming warning: two different papers are called "DeepCrack". The 537-image dataset is from Liu et al. (Neurocomputing, 2019), not Zou et al. (IEEE Transactions on Image Processing, 2019). Cite the right one.
- Image sizes are roughly 384 to 544 px per side, so training crop size is set from measured image sizes, not assumed.
- Validation images are carved out of the official training split.

**BD3 (hard negatives)**
- Reported by its authors: 3,965 phone photos from campus walls at IISc, at about 1 m distance, across 50+ buildings. Classes: algae 624, major crack 620, minor crack 580, peeling 520, spalling 500, stain 521, normal 600. The authors provide a 60/20/20 split.
- To verify: license (none seen on the page used when planning this project).
- Use: only the **non-crack classes** (stain, peeling, algae, spalling, normal) become all-zero masks, sampled at roughly 20 to 30% of training crops.
- Quality control: before use, spot-check 30 images per class and **drop any that contain a crack**. Spalling images in particular may show cracks.
- BD3 has many near-duplicate shots per building, so a random split would leak. BD3 is used for training negatives only.

**ConViD (stress test)**
- Reported: phone photos from Pune, 12 MP square, classes honeycombing, voids, cracks and spalling, captured in morning, afternoon and night light; license CC BY 4.0 per its Mendeley page.
- To verify: image counts. Its authors note heavy class overlap, so expect label noise.
- Use: test only. Non-crack classes measure the false-alarm rate. About 25 crack images are **hand-annotated** by the project author to give a real-world IoU and thin-line score.
- **Source separation:** BD3 supplies training negatives and ConViD supplies the test, so near-duplicate images cannot leak from training into test.

### Datasets considered and excluded

| Dataset | Why excluded |
|---|---|
| CrackSeg9k, Khanh11k / Kaggle11k, Conglomerate | Merged collections that include Crack500, DeepCrack and others. They would contaminate any held-out set from those families. Published analysis also found duplicate images across such collections leaking between train and test. |
| OmniCrack30k as a training source | Same merging problem. Only its **metric definition** is used. |
| "Concrete Crack images for segmentation" (Tongji) | Partly built from Özgenel and DeepCrack images; would contaminate the held-out set. |
| Özgenel 40,000-patch classification dataset | Cut from the same 458 photos as the held-out set. |
| MBDD2025 | Drone imagery for object detection; wrong scale for hairline cracks and an unverified annotation format. |
| Cracks in the Foundation | Too hard and too heavy (mostly European bridges) for this project's scope. |

### Mask convention

All masks are converted to one format: PNG, `0` = background, `1` = crack. Source datasets differ (white-on-black versus black-on-white, 0/255 versus 0/1, JPEG-stored masks with in-between values that must be thresholded). A conversion script and a test assert that only 0 and 1 appear, and crack pixels should be a few percent of all pixels. A much larger share signals flipped polarity.

Datasets themselves are **not** redistributed in this repository. `DATA.md` records sources, versions, license status and verification results.

## Evaluation protocol

These rules are fixed before any model is trained.

1. **The held-out set is locked first.** Its file list is saved in `splits/` with a fingerprint (hash) per file. The training code **refuses to start** if any held-out file appears among its inputs, and a test verifies this by deliberately injecting a held-out file.
2. **Choices are made on validation data only.** This covers the cut-off, the number of training epochs, and the settings of the classical baseline.
3. **The held-out set is read, not tuned on.** It is evaluated once per finished model. No change to the training setup is justified by held-out scores. If a change is made after looking, it is logged as such.
4. **Splits respect the original photo.** Crops of one photo never straddle train and test. Cropping happens after splitting.
5. **Cut-off is frozen** after validation and used unchanged on every other dataset.
6. **Report distributions, not just means.** Every per-image score is saved. Tables show median and spread (for example worst and best 10%), and failures are inspected visually.
7. **Variance is measured.** Two to three training seeds are used where compute allows, and the number of seeds behind every number is stated. A single run cannot separate a real difference from luck, and results from one run are labelled as such.
8. **Always state the tolerance** next to a thin-line score.
9. **Pixel-width "truth" is mask truth.** Width error is measured against widths computed from the ground-truth masks, which are human-drawn. This validates the geometry module, not physical millimetres.

## Metrics

All metrics are implemented from scratch in `src/` with unit tests. Libraries are not used for the metric logic.

**IoU** = `|P ∩ T| / |P ∪ T|`, where P is the predicted crack pixels and T the true crack pixels.

**Dice** = `2 |P ∩ T| / (|P| + |T|)`.

**Pixel loss** = binary cross-entropy (BCE): a pixel-by-pixel penalty for wrong probabilities. Cracks are only a few percent of pixels, so a model that says "no crack anywhere" is already mostly right; the overlap term below prevents that shortcut.

**Training loss** = BCE + Dice loss, where the Dice term operates on predicted probabilities rather than hard 0/1 decisions so it can be differentiated.

**Thin-line score (centerline IoU, tolerance τ = 4 px)**, following the OmniCrack30k benchmark definition:
- Thin the true mask to a skeleton `S_T` and the predicted mask to `S_P` (Guo–Hall thinning).
- `TP` = true-skeleton pixels lying within τ of the predicted skeleton.
- `FP` = predicted-skeleton pixels lying farther than τ from the true skeleton.
- `FN` = true-skeleton pixels with no predicted skeleton within τ.
- Score = `TP / (TP + FP + FN)`.
- Note the asymmetry: true positives are counted on the true skeleton and false positives on the predicted skeleton. Scores depend strongly on τ (much lower at 0 px than at 4 px), so τ is always reported.

**Edge cases.** The behaviour when both prediction and truth are empty (0 ÷ 0) is an explicit, documented decision in the metrics module and is covered by a test. It matters for crack-free images.

**False-alarm rate** (on crack-free images) is reported both at image level (does the model output a predicted component above a minimum size?) and as the share of pixels wrongly marked. The minimum size is fixed in config.

## Models

- **Classical baseline (no learning).** Dark-thin-line detection (black-hat filtering) or Otsu thresholding, followed by morphological clean-up. Settings are tuned on validation data only. This is the floor: if the learned model barely beats it on held-out data, that is a finding.
- **U-Net with a pretrained encoder**, built with `segmentation_models_pytorch`. Default encoder: ImageNet-pretrained ResNet-34 (configurable; the reason for the final choice is recorded in the config). The pretrained encoder expects images normalised the way its original training images were, and the project applies the matching preprocessing.
- **Training:** BCE + Dice loss on fixed-size square crops. Crop size is set from measured dataset image sizes. Augmentation is deliberately simple: flips, 90° rotations, small brightness changes.
- **Using a library model here is intentional.** The project's value is in the evaluation and system, not in re-implementing a network.

## Inference on large images

Held-out photos are roughly 12 megapixels, far larger than the network's training crops, and hairline cracks vanish if the image is downscaled to fit. CrackLens instead predicts at **native resolution**:

1. Cut the image into overlapping tiles (tile size equals the training crop size).
2. Predict each tile.
3. Average predictions where tiles overlap, which reduces seams at tile borders where the network has less context.
4. Apply the frozen cut-off.

Tests check that stitched output has the right shape, that a constant dummy model yields a constant result, and that predictions barely change when the tile size changes. Overlap is configurable and is chosen by inspecting seams and by runtime on the largest images.

## Geometry module

Computed from a binary mask, with no scale needed. All quantities are in pixels or unitless ratios.

| Statistic | Definition |
|---|---|
| `width_px_max`, `width_px_p95` | Skeletonise the mask, sample the distance transform along the skeleton, and take twice that distance as local width. Report the maximum and 95th percentile. |
| `relative_width` | `width_px / min(image height, image width)` (computed for the max and p95 widths). |
| `area_ratio` | Crack pixels divided by total pixels. |
| `length_ratio` | Skeleton pixel count divided by the image diagonal (an approximate length). |
| `component_count` | Number of connected crack components (8-connectivity), after discarding components below a minimum size set in config. |
| `branch_point_count` | Number of skeleton junctions (pixels with three or more skeleton neighbours, adjacent junction pixels merged). |

If `px_per_mm` is supplied, width and length are additionally returned in millimetres (`mm = px / px_per_mm`). Ratios and counts are unchanged.

Distance-transform width is used because comparable published work found it the most accurate of several mask-based width estimators, and because it is simple to test. It is expected to be unreliable for very thin cracks, where a mask only a pixel or two wide leaves little room for a stable estimate, and published results on thin cracks show widths being over-measured rather than under-measured. Whether the same direction of error holds here is checked, not assumed.

**Unit tests** use synthetic masks of known width (for example 1, 2, 3, 5 and 10 px; straight, curved and branching), plus a test for the `px_per_mm` conversion.

## Experiments

Each experiment states a hypothesis, a baseline, the metric, and the single variable changed.

1. **Cross-dataset gap (headline).** Classical vs U-Net, scored on the in-domain tests (Crack500 and DeepCrack separately) and on the held-out set. Per-image distributions reported. At least two causes are mixed in the gap (different surfaces, and different crack sizes in pixels), so a cause is not claimed without a test; the resolution experiment below probes the second.
2. **Hard-negative ablation.** Train with and without BD3 negatives, everything else identical. Compare false-alarm rate on ConViD's crack-free images and IoU on the held-out set. Repeated over seeds where possible; the number of seeds is stated.
3. **Real-world stress test.** On ConViD: false-alarm rate on non-crack classes, and IoU plus thin-line score on the hand-annotated crack images, with and without negatives.
4. **Width-error study.** Compute geometry on ground-truth masks and on predicted masks, and compare. Error is binned by true width. Failure for 1 to 3 px cracks is expected and reported as a finding. Direction of the error (over- or under-measurement) is reported.
5. **Resolution experiment.** Downscale test images by several factors and plot recall on thin cracks (a fixed definition of "thin", for example ground-truth width of 1 to 3 px at native resolution, is set before running) against the factor.
6. **Error analysis.** A failure gallery with categories: shadows, joints and seams, stains, thin cracks. Best and worst images by score are inspected visually.

Stretch only: pseudo-labels for image-level-only data. Never scored on pseudo-labels.

## Serving

Planned interface (the final shape is documented here once implemented):

- `POST /analyze`: multipart image upload, optional `px_per_mm`. Returns a mask PNG and a JSON of statistics. The JSON states its units (`px` or `mm`) explicitly.
- `GET /health`

The response schema is **stable and versioned**, so findings can be stored and queried by other systems.

```
{
  "schema_version": <string>,
  "image":   { "height_px": <int>, "width_px": <int> },
  "units":   "px" | "mm",
  "stats": {
    "width_px_max": <float>, "width_px_p95": <float>,
    "relative_width_max": <float>, "relative_width_p95": <float>,
    "area_ratio": <float>, "length_ratio": <float>,
    "component_count": <int>, "branch_point_count": <int>,
    "width_mm_max": <float | null>, "width_mm_p95": <float | null>, "length_mm": <float | null>
  },
  "mask_png_base64": <string>
}
```

Deployment components:
- **ONNX export**, with a parity check against the PyTorch model's outputs.
- **CPU latency** reported as p50 and p95 per tile.
- **Dockerfile** producing a container that runs the service.
- **GitHub Actions CI** that runs the metric and geometry tests on every push.

The API is **synchronous**: large images block the request. A job queue is out of scope and this is a documented limitation.

## Testing

| Test | What it protects |
|---|---|
| Metric tests: perfect match, empty prediction vs non-empty truth, both empty, hand-computed partial overlap | Correct scoring |
| Thin-line score tests: parallel lines within and beyond tolerance, empty cases | Correct tolerance logic |
| Mask-format test: only 0 and 1 present | Consistent data |
| Held-out guard test: a held-out file injected into training must stop the run | No leakage into training |
| Geometry tests on synthetic masks of known width, and the `px_per_mm` conversion | Correct geometry |
| Tiling tests: shape, constant model, tile-size consistency | Correct stitching |
| End-to-end sanity: feeding ground truth as the prediction gives 1.0 on every score | Whole scoring chain |

CI runs the metric and geometry tests.

## Reproducibility and experiment discipline

- Every run reads a **config file** (crop size, batch size, learning rate, epochs, encoder, seed, cut-off source).
- Seeds are fixed and recorded.
- Each run appends a row to a **results CSV** with config identifier, dataset versions, scores and seed.
- Per-image scores are saved so distributions can be recomputed.
- Dataset versions and sources are recorded in `DATA.md`.
- Training is designed to run unattended: checkpointing, best checkpoint chosen by validation score, results written at the end.
- Experiment tracking with MLflow is used locally only if it is free to set up.

## Repository layout

```
README.md          this file
DATA.md            dataset sources, versions, licenses, verification results
configs/           run configurations
src/               metrics, data handling, models, tiling, geometry, evaluation, serving
tests/             unit tests (metrics, geometry, tiling, guards)
splits/            locked held-out file list + fingerprints; other split lists
results/           results CSVs, per-image scores, figures, failure gallery
data/              datasets (not tracked in git)
Dockerfile         service container            (added with serving)
.github/workflows/ CI that runs the tests       (added with serving)
```

## Results

**No number appears in this section until it has been produced by the code in this repository.** Tables are filled from the saved results CSVs, with the number of seeds stated next to every figure.

| Table | Contents | Status |
|---|---|---|
| Cross-dataset gap | Classical vs U-Net; Crack500 test, DeepCrack test, held-out set; IoU, Dice, thin-line score; median and spread | Not yet measured |
| Hard-negative ablation | With vs without negatives; false-alarm rate; held-out IoU | Not yet measured |
| Stress tests | ConViD false-alarm rate; IoU and thin-line score on hand-annotated set | Not yet measured |
| Width error | Error binned by true width, with direction of error | Not yet measured |
| Resolution | Thin-crack recall vs downscale factor | Not yet measured |
| Latency | ONNX CPU p50 / p95 per tile | Not yet measured |
| Failure gallery | Shadows, joints, stains, thin cracks | Not yet produced |

## Limitations

These apply to the system as built and are repeated in any write-up of results.

- **No scale.** Without `px_per_mm`, no physical size is reported. Even with it, accuracy depends entirely on the supplied value being correct for that photo, and it is only meaningful for roughly fronto-parallel surfaces.
- **Pavement-heavy training data.** Both training datasets lean on pavement or mixed surfaces, while the target domain is building concrete.
- **Single-campus held-out set.** The held-out result shows generalisation to one other set of buildings, not to all concrete.
- **Annotation is human.** Ground-truth masks depend on the annotator's line thickness and judgement. Width error is measured against these masks, not against physical measurements. The hand-annotated ConViD set is annotated by a single person.
- **Thin cracks are hard.** Cracks one to three pixels wide are expected to give poor width estimates and lower scores, and are reported as such.
- **Label noise in image-level datasets.** BD3 and ConViD labels were not made for segmentation; BD3 negatives are spot-checked but not exhaustively verified, and ConViD has overlapping classes.
- **Some dataset terms are restrictive or unverified.** See [Licenses](#licenses-data-terms-and-citations).
- **Synchronous API.** Large images are processed within the request.
- **Not a safety tool.** Outputs must not be used for structural-safety decisions.

## Questions this project is built to answer

1. Why segmentation rather than detection or classification?
2. Why this encoder, and what was it chosen against?
3. How big is the cross-dataset gap, and what plausible causes can and cannot be separated?
4. What do hard negatives change, and how do we know the change is real rather than noise?
5. Why can't the system report millimetres by default, and what would it take?
6. What happens to hairline cracks when an image is downscaled?
7. How was leakage between the negative-source and test-source datasets prevented?
8. What does the width error look like for 1 to 3 px cracks, and in which direction does it fail?
9. How does this differ from published end-to-end crack-assessment work that calibrates to millimetres and scores severity? (CrackLens deliberately does less, and measures it more carefully.)

## Definition of done

- [ ] Held-out result table (classical vs U-Net, in-domain vs held-out)
- [ ] Width-error table against ground-truth masks
- [ ] Failure gallery
- [ ] Stress-test table (hard-negative ablation and ConViD results)
- [ ] Resolution experiment plot
- [ ] ONNX latency numbers (p50 and p95 per tile)
- [ ] Runnable container
- [ ] CI running the metric and geometry tests
- [ ] This README with results filled in and limitations current

## Licenses, data terms and citations

**Code license:** to be chosen before publishing.

**Data is not redistributed here.** Obtain each dataset from its source and follow its terms.

| Dataset | Terms (status) |
|---|---|
| Özgenel segmentation dataset | CC BY 4.0 (verified on the dataset page). Attribution required. |
| DeepCrack (Liu et al.) | Non-commercial research and education only, per the authors (verified). Do not redistribute images or annotations. |
| Crack500 | No license stated in the authors' repository README (verified absence); citation requested. Treat as research use; unverified. |
| BD3 | License unverified. |
| ConViD | CC BY 4.0 per its Mendeley page; confirm when downloading. |

Models trained on non-commercial data inherit that context; state the terms alongside any released weights and check them before any commercial use.

**Citations**

- Özgenel, Ç. F. (2019). *Concrete Crack Segmentation Dataset.* Mendeley Data, V1. doi:10.17632/jwsn7tfbrp.1
- Yang, F., Zhang, L., Yu, S., Prokhorov, D., Mei, X., Ling, H. (2020). Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection. *IEEE Transactions on Intelligent Transportation Systems*, 21(4), 1525–1535. And Zhang, L., Yang, F., Zhang, Y. D., Zhu, Y. J. (2016). Road crack detection using deep convolutional neural network. *ICIP 2016.*
- Liu, Y., Yao, J., Lu, X., Xie, R., Li, L. (2019). DeepCrack: A deep hierarchical feature learning architecture for crack segmentation. *Neurocomputing*, 338, 139–153.
- Benz, C., Rodehorst, V. (2024). OmniCrack30k: A Benchmark for Crack Segmentation and the Reasonable Effectiveness of Transfer Learning. *CVPR Workshops 2024.* (Source of the thin-line score definition.)
- Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015.*
- Rostami, G., Chen, P.-H., Hosseini, M. S. (2026). Inspect Any Crack: An Automated End-to-End Crack Segmentation and Severity Assessment Framework. arXiv:2610.14274. (Closest related work; preprint.)
- Iakubovskii, P. *Segmentation Models PyTorch.* GitHub, 2019.
- BD3 and ConViD: cite as specified on their dataset pages.
