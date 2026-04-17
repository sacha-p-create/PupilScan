// Copyright 2026 [Your Name].
//
// Adapted from StrabScan by Rahul Dhodapkar, which was itself
// adapted from content released by The MediaPipe Authors under
// the Apache License, Version 2.0.
//
// Released under GNU GPL v3 (LICENSE) in accordance with the
// license file included in this directory.
//
// Changes from StrabScan:
//   - Compute iris center as centroid of 4 ring landmarks (469-472, 474-477)
//     rather than trusting MediaPipe's single landmark 468/473 directly
//   - Fix coordinate space mismatch: iris diameter now computed in normalized
//     space using vecNorm (rotation-invariant) instead of raw x-axis span
//   - Add exponential moving average smoothing on iris center positions
//     to reduce frame-to-frame landmark jitter

console.log("starting load")

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3"
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision
const demosSection = document.getElementById("demos")
const imageBlendShapes = document.getElementById("image-blend-shapes")
const videoBlendShapes = document.getElementById("video-blend-shapes")

// ===========================================================
// =========== DEFINE GLOBAL CONSTANTS =======================
// ===========================================================

let faceLandmarker
let runningMode = "IMAGE"
let enableWebcamButton
let webcamRunning = false
const videoWidth = 480

let currentFacingMode = "environment"; // "user" (front) | "environment" (back)
let currentStream = null;

let currentLightOnStatus = false;

let currentOverlayVisibleStatus = true;

let PREDICTED_DEVIATION_SCALE_FACTOR = 0.34641
let MAX_NUM_DEVIATION_OBSERVATIONS = 500

let qualityControlPassedDeviationResults = []

// right eye landmarks
let MP_RT_IRIS_CENTER = 468
let MP_RT_IRIS_MED = 469 // (medial)
let MP_RT_IRIS_SUP = 470 // (superior)
let MP_RT_IRIS_LAT = 471 // (lateral)
let MP_RT_IRIS_INF = 472 // (inferior)

let MP_RT_LAT_CANTHUS = 33
let MP_RT_MED_CANTHUS = 133

// left eye landmarks
let MP_LT_IRIS_CENTER = 473
let MP_LT_IRIS_MED = 476 // (medial)
let MP_LT_IRIS_SUP = 475 // (superior)
let MP_LT_IRIS_LAT = 474 // (lateral)
let MP_LT_IRIS_INF = 477 // (inferior)

let MP_LT_LAT_CANTHUS = 263
let MP_LT_MED_CANTHUS = 362

//midface landmarks - may be useful for sanity testing
let MP_MIDPOINT_BETWEEN_EYES = 168
let MP_NOSE_TIP = 1
let MP_NOSE_BOTTOM = 2
let MP_FOREHEAD_TOP = 10
let MP_CHIN_BOTTOM = 152
let MP_LIP_CENTER = 0

let MP_RT_LIP_EDGE = 61
let MP_LT_LIP_EDGE = 291

let MP_RT_EAR = 234
let MP_LT_EAR = 454

// ===========================================================
// =========== TEMPORAL SMOOTHING STATE ======================
// ===========================================================

// Exponential moving average (EMA) smoothing for iris centers.
// ALPHA controls responsiveness vs stability:
//   higher (e.g. 0.5) = reacts faster but noisier
//   lower  (e.g. 0.1) = smoother but slightly lags fast movement
// 0.2 is a good starting point for strabismus measurement.
const SMOOTHING_ALPHA = 0.2;

let smoothedRtCenter = null;
let smoothedLtCenter = null;

function smoothPoint(prev, curr, alpha) {
  if (!prev) return curr;
  return [
    alpha * curr[0] + (1 - alpha) * prev[0],
    alpha * curr[1] + (1 - alpha) * prev[1],
    alpha * curr[2] + (1 - alpha) * prev[2],
  ];
}

// Call this when the webcam is stopped/restarted so smoothing
// doesn't carry over stale state from a previous session.
function resetSmoothing() {
  smoothedRtCenter = null;
  smoothedLtCenter = null;
}

console.log("===== loaded constants successfully =====")

// ===========================================================
// =========== INCLUDE BARYCENTRIC MODEL STRINGS =============
// ===========================================================

let bary_model_rt_string = `w_value,selected
0.0,0
0.0,1
0.0,2
0.0,3
0.0,4
0.0,5
0.0,6
0.0,7
0.0,8
0.0,9
0.0,10
0.0,11
0.0,12
0.0,13
0.0,14
0.0,15
0.0,16
0.0,17
0.0,18
0.0,19
0.0,20
0.024062760774899335,21
0.0,22
0.0,23
0.0,24
0.0,25
0.0,26
0.0,27
0.0,28
0.0,29
0.0,30
0.0,31
0.0,32
0.3233774305751091,33
0.0,34
0.0,35
0.0,36
0.0,37
0.0,38
0.0,39
0.0,40
0.0,41
0.0,42
0.0,43
0.0,44
0.0,45
0.0,46
0.0,47
0.0,48
0.0,49
0.0,50
0.0,51
0.0,52
0.0,53
0.0005552165602942576,54
0.0,55
0.0,56
0.0,57
0.0,58
0.0,59
0.0,60
0.0,61
0.0,62
0.0,63
0.0,64
0.0,65
0.0,66
0.0,67
0.0,68
0.0,69
0.0,70
0.0,71
0.0,72
0.0,73
0.0,74
0.0,75
0.0,76
0.0,77
0.0,78
0.0,79
0.0,80
0.0,81
0.0,82
0.0,83
0.0,84
0.0,85
0.0,86
0.0,87
0.0,88
0.0,89
0.0,90
0.0,91
0.0,92
0.0,93
0.0,94
0.0,95
0.0,96
0.0,97
0.0,98
0.0,99
0.0,100
0.0,101
0.0,102
0.0,103
0.0,104
0.0,105
0.0,106
0.0,107
0.0,108
0.0,109
0.0,110
0.0,111
0.0,112
0.0,113
0.0,114
0.0,115
0.0,116
0.0,117
0.0,118
0.0,119
0.0,120
0.0,121
0.0,122
0.0,123
0.0,124
0.0,125
0.0,126
0.0,127
0.0,128
0.0,129
0.0,130
0.0,131
0.0,132
0.0,133
0.0,134
0.0,135
0.0,136
0.0,137
0.0,138
0.0,139
0.0,140
0.0,141
0.0,142
0.0,143
0.0,144
0.1138583116055262,145
0.0,146
0.0,147
0.0,148
0.0,149
0.0,150
0.0,151
0.0,152
0.1228656188388233,153
0.0,154
0.0,155
0.0,156
0.3202614773449372,157
0.08389602258771495,158
0.0,159
0.0,160
0.0,161
0.0,162
0.0,163
0.0,164
0.0,165
0.0,166
0.0,167
0.0,168
0.0,169
0.0,170
0.0,171
0.0,172
0.0,173
0.0,174
0.0,175
0.0,176
0.0,177
0.0,178
0.0,179
0.0,180
0.0,181
0.0,182
0.0,183
0.0,184
0.0,185
0.0,186
0.0,187
0.0,188
0.0,189
0.0,190
0.0,191
0.0,192
0.0,193
0.0,194
0.0,195
0.0,196
0.0,197
0.0,198
0.0,199
0.0,200
0.0,201
0.0,202
0.0,203
0.0,204
0.0,205
0.0,206
0.0,207
0.0,208
0.0,209
0.0,210
0.0,211
0.0,212
0.0,213
0.0,214
0.0,215
0.0,216
0.0,217
0.0,218
0.0,219
0.0,220
0.0,221
0.0,222
0.0,223
0.0,224
0.0,225
0.0,226
0.0,227
0.0,228
0.0,229
0.0,230
0.0,231
0.0,232
0.0,233
0.0,234
0.0,235
0.0,236
0.0,237
0.0,238
0.0,239
0.0,240
0.0,241
0.0,242
0.0,243
0.0,244
0.0,245
0.0,246
0.0,247
0.0,248
0.0,249
0.0,250
0.0,251
0.0,252
0.0,253
0.0,254
0.0,255
0.0,256
0.0,257
0.0,258
0.0,259
0.0,260
0.0,261
0.0,262
0.0,263
0.0,264
0.0,265
0.0,266
0.0,267
0.0,268
0.0,269
0.0,270
0.0,271
0.0,272
0.0,273
0.0,274
0.0,275
0.0,276
0.0,277
0.0,278
0.0,279
0.0,280
0.0,281
0.0,282
0.0,283
0.0,284
0.0,285
0.0,286
0.0,287
0.0,288
0.0,289
0.0,290
0.0,291
0.0,292
0.0,293
0.0,294
0.0,295
0.0,296
0.0,297
0.0,298
0.0,299
0.0,300
0.0,301
0.0,302
0.0,303
0.0,304
0.0,305
0.0,306
0.0,307
0.0,308
0.0,309
0.0,310
0.0,311
0.0,312
0.0,313
0.0,314
0.0,315
0.0,316
0.0,317
0.0,318
0.0,319
0.0,320
0.0,321
0.0,322
0.0,323
0.0,324
0.0,325
0.0,326
0.0,327
0.0,328
0.0,329
0.0,330
0.0,331
0.0,332
0.0,333
0.0,334
0.0,335
0.0,336
0.0,337
0.0,338
0.0,339
0.0,340
0.0,341
0.0,342
0.0,343
0.0,344
0.0,345
0.0,346
0.0,347
0.0,348
0.0,349
0.0,350
0.0,351
0.0,352
0.0,353
0.0,354
0.0,355
0.011123161712695572,356
0.0,357
0.0,358
0.0,359
0.0,360
0.0,361
0.0,362
0.0,363
0.0,364
0.0,365
0.0,366
0.0,367
0.0,368
0.0,369
0.0,370
0.0,371
0.0,372
0.0,373
0.0,374
0.0,375
0.0,376
0.0,377
0.0,378
0.0,379
0.0,380
0.0,381
0.0,382
0.0,383
0.0,384
0.0,385
0.0,386
0.0,387
0.0,388
0.0,389
0.0,390
0.0,391
0.0,392
0.0,393
0.0,394
0.0,395
0.0,396
0.0,397
0.0,398
0.0,399
0.0,400
0.0,401
0.0,402
0.0,403
0.0,404
0.0,405
0.0,406
0.0,407
0.0,408
0.0,409
0.0,410
0.0,411
0.0,412
0.0,413
0.0,414
0.0,415
0.0,416
0.0,417
0.0,418
0.0,419
0.0,420
0.0,421
0.0,422
0.0,423
0.0,424
0.0,425
0.0,426
0.0,427
0.0,428
0.0,429
0.0,430
0.0,431
0.0,432
0.0,433
0.0,434
0.0,435
0.0,436
0.0,437
0.0,438
0.0,439
0.0,440
0.0,441
0.0,442
0.0,443
0.0,444
0.0,445
0.0,446
0.0,447
0.0,448
0.0,449
0.0,450
0.0,451
0.0,452
0.0,453
0.0,454
0.0,455
0.0,456
0.0,457
0.0,458
0.0,459
0.0,460
0.0,461
0.0,462
0.0,463
0.0,464
0.0,465
0.0,466
0.0,467
0.0,473
0.0,476
0.0,475
0.0,474
0.0,477`

let bary_model_lt_string = `w_value,selected
0.0,0
0.0,1
0.0,2
0.0,3
0.0,4
0.0,5
0.0,6
0.0,7
0.0,8
0.0,9
0.0,10
0.0,11
0.0,12
0.0,13
0.0,14
0.0,15
0.0,16
0.0,17
0.0,18
0.0,19
0.0,20
0.0,21
0.0,22
0.0,23
0.0,24
0.0,25
0.0,26
0.0,27
0.0,28
0.0,29
0.0,30
0.0,31
0.0,32
0.0,33
0.0,34
0.0,35
0.0,36
0.0,37
0.0,38
0.0,39
0.0,40
0.0,41
0.0,42
0.0,43
0.0,44
0.0,45
0.0,46
0.0,47
0.0,48
0.0,49
0.0,50
0.0,51
0.0,52
0.0,53
0.0,54
0.0,55
0.0,56
0.0,57
0.0,58
0.0,59
0.0,60
0.0,61
0.0,62
0.0,63
0.0,64
0.0,65
0.0,66
0.0,67
0.0,68
0.0,69
0.0,70
0.0,71
0.0,72
0.0,73
0.0,74
0.0,75
0.0,76
0.0,77
0.0,78
0.0,79
0.0,80
0.0,81
0.0,82
0.0,83
0.0,84
0.0,85
0.0,86
0.0,87
0.0,88
0.0,89
0.0,90
0.0,91
0.0,92
0.0,93
0.0,94
0.0,95
0.0,96
0.0,97
0.0,98
0.0,99
0.0,100
0.0,101
0.0,102
0.0,103
0.0,104
0.0,105
0.0,106
0.0,107
0.0,108
0.0,109
0.0,110
0.0,111
0.0,112
0.0,113
0.0,114
0.0,115
0.0,116
0.0,117
0.0,118
0.0,119
0.0,120
0.0,121
0.0,122
0.0,123
0.0,124
0.0,125
0.0,126
0.014158980788587768,127
0.0,128
0.0,129
0.0,130
0.0,131
0.0,132
0.0,133
0.0,134
0.0,135
0.0,136
0.0,137
0.0,138
0.0,139
0.0,140
0.0,141
0.0,142
0.0,143
0.0,144
0.0,145
0.0,146
0.0,147
0.0,148
0.0,149
0.0,150
0.0,151
0.0,152
0.0,153
0.0,154
0.0,155
0.0,156
0.0,157
0.0,158
0.0,159
0.0,160
0.0,161
0.0,162
0.0,163
0.0,164
0.0,165
0.0,166
0.0,167
0.0,168
0.0,169
0.0,170
0.0,171
0.0,172
0.0,173
0.0,174
0.0,175
0.0,176
0.0,177
0.0,178
0.0,179
0.0,180
0.0,181
0.0,182
0.0,183
0.0,184
0.0,185
0.0,186
0.0,187
0.0,188
0.0,189
0.0,190
0.0,191
0.0,192
0.0,193
0.0,194
0.0,195
0.0,196
0.0,197
0.0,198
0.0,199
0.0,200
0.0,201
0.0,202
0.0,203
0.0,204
0.0,205
0.0,206
0.0,207
0.0,208
0.0,209
0.0,210
0.0,211
0.0,212
0.0,213
0.0,214
0.0,215
0.0,216
0.0,217
0.0,218
0.0,219
0.0,220
0.0,221
0.0,222
0.0,223
0.0,224
0.0,225
0.0,226
0.0,227
0.0,228
0.0,229
0.0,230
0.0,231
0.0,232
0.0,233
0.0,234
0.0,235
0.0,236
0.0,237
0.0,238
0.0,239
0.0,240
0.0,241
0.0,242
0.0,243
0.0,244
0.0,245
0.0,246
0.0,247
0.0,248
0.0,249
0.0,250
0.006359732068736367,251
0.0,252
0.0,253
0.0,254
0.0,255
0.0,256
0.0,257
0.0,258
0.0,259
0.0,260
0.0,261
0.0,262
0.25896051270249176,263
0.0,264
0.0,265
0.0,266
0.0,267
0.0,268
0.0,269
0.0,270
0.0,271
0.0,272
0.0,273
0.0,274
0.0,275
0.0,276
0.0,277
0.0,278
0.0,279
0.0,280
0.0,281
0.0,282
0.0,283
0.01628764562411962,284
0.0,285
0.0,286
0.0,287
0.0,288
0.0,289
0.0,290
0.0,291
0.0,292
0.0,293
0.0,294
0.0,295
0.0,296
0.0,297
0.0,298
0.0,299
0.0,300
0.0,301
0.0,302
0.0,303
0.0,304
0.0,305
0.0,306
0.0,307
0.0,308
0.0,309
0.0,310
0.0,311
0.0,312
0.0,313
0.0,314
0.0,315
0.0,316
0.0,317
0.0037480361975083597,318
0.0,319
0.0,320
0.0,321
0.0,322
0.0,323
0.0,324
0.0,325
0.0,326
0.0,327
0.0,328
0.0,329
0.0,330
0.0,331
0.0,332
0.0,333
0.0,334
0.0,335
0.0,336
0.0,337
0.0,338
0.0,339
0.0,340
0.0,341
0.0,342
0.0,343
0.0,344
0.0,345
0.0,346
0.0,347
0.0,348
0.0,349
0.0,350
0.0,351
0.0,352
0.0,353
0.0,354
0.0,355
0.0,356
0.0,357
0.0,358
0.0,359
0.0,360
0.0,361
0.0,362
0.0,363
0.0,364
0.0,365
0.0,366
0.0,367
0.0,368
0.0,369
0.0,370
0.0,371
0.0,372
0.0,373
0.10784192671506068,374
0.0,375
0.0,376
0.0,377
0.0,378
0.0,379
0.21382920912908382,380
0.0,381
0.0,382
0.0,383
1.5627082668982565e-17,384
0.37881395677441165,385
0.0,386
0.0,387
0.0,388
0.0,389
0.0,390
0.0,391
0.0,392
0.0,393
0.0,394
0.0,395
0.0,396
0.0,397
0.0,398
0.0,399
0.0,400
0.0,401
0.0,402
0.0,403
0.0,404
0.0,405
0.0,406
0.0,407
0.0,408
0.0,409
0.0,410
0.0,411
0.0,412
0.0,413
0.0,414
0.0,415
0.0,416
0.0,417
0.0,418
0.0,419
0.0,420
0.0,421
0.0,422
0.0,423
0.0,424
0.0,425
0.0,426
0.0,427
0.0,428
0.0,429
0.0,430
0.0,431
0.0,432
0.0,433
0.0,434
0.0,435
0.0,436
0.0,437
0.0,438
0.0,439
0.0,440
0.0,441
0.0,442
0.0,443
0.0,444
0.0,445
0.0,446
0.0,447
0.0,448
0.0,449
0.0,450
0.0,451
0.0,452
0.0,453
0.0,454
0.0,455
0.0,456
0.0,457
0.0,458
0.0,459
0.0,460
0.0,461
0.0,462
0.0,463
0.0,464
0.0,465
0.0,466
0.0,467
0.0,468
0.0,469
0.0,470
0.0,471
0.0,472`

const bary_model_rt = Papa.parse(bary_model_rt_string, { header: true })
const bary_model_lt = Papa.parse(bary_model_lt_string, { header: true })

console.log("===== loaded barycentric models successfully =====")

// ===========================================================
// =========== SCRIPTS FOR STRAB MEASUREMENT =================
// ===========================================================

function vecAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecScale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vecDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vecCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function vecNorm(v) {
  return Math.sqrt(vecDot(v, v));
}

function vecNormalize(v) {
  const n = vecNorm(v);
  return n > 1e-8 ? vecScale(v, 1 / n) : [0, 0, 0];
}

function matVecMul(v, M) {
  return [
    v[0] * M[0][0] + v[1] * M[1][0] + v[2] * M[2][0],
    v[0] * M[0][1] + v[1] * M[1][1] + v[2] * M[2][1],
    v[0] * M[0][2] + v[1] * M[1][2] + v[2] * M[2][2],
  ];
}

function matTranspose(M) {
  return [
    [M[0][0], M[1][0], M[2][0]],
    [M[0][1], M[1][1], M[2][1]],
    [M[0][2], M[1][2], M[2][2]],
  ];
}

function facemeshToTensor(landmarks) {
  return landmarks.map(lm => [lm.x, lm.y, lm.z]);
}

function alignmentMatrix(
  landmarks,
  outerLeftIdx = 263,
  outerRightIdx = 33,
  noseTipIdx = 1
) {
  const X = facemeshToTensor(landmarks);

  const leftEye = X[outerLeftIdx];
  const rightEye = X[outerRightIdx];

  const zAxis = [0, 0, 1];

  let xAxis = vecSub(rightEye, leftEye);
  const proj = vecScale(zAxis, vecDot(xAxis, zAxis));
  xAxis = vecNormalize(vecSub(xAxis, proj));

  let yAxis = vecNormalize(vecCross(xAxis, zAxis));

  const R = [
    [xAxis[0], yAxis[0], zAxis[0]],
    [xAxis[1], yAxis[1], zAxis[1]],
    [xAxis[2], yAxis[2], zAxis[2]],
  ];

  const eyeDist = vecNorm(vecSub(rightEye, leftEye));
  const s = eyeDist > 1e-6 ? 1.0 / eyeDist : 1.0;

  const mean = X.reduce(
    (acc, p) => vecAdd(acc, p),
    [0, 0, 0]
  ).map(v => v / X.length);

  const meanRot = matVecMul(mean, R);
  const t = vecScale(meanRot, -s);

  return { R, s, t };
}

function applyAlignment(landmarks, R, s, t) {
  const X = facemeshToTensor(landmarks);
  return X.map(p => {
    const rotated = matVecMul(p, R);
    return vecAdd(vecScale(rotated, s), t);
  });
}

function reverseAlignment(points, R, s, t) {
  const Rt = matTranspose(R);
  return points.map(p => {
    const unscaled = vecScale(vecSub(p, t), 1 / s);
    return matVecMul(unscaled, Rt);
  });
}

function predictBarycentricPoint(weights, indices, normalized) {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < weights.length; i++) {
    const w = parseFloat(weights[i]);
    const p = normalized[parseInt(indices[i])];
    x += w * p[0];
    y += w * p[1];
    z += w * p[2];
  }
  return [x, y, z];
}

function degreesToPD(deg) {
  return Math.tan(deg * Math.PI / 180) * 100;
}

// FIX 1: Compute iris center as centroid of the 4 ring landmarks.
// Previously the code used MediaPipe's single landmark 468/473 directly.
// Averaging all 4 ring points reduces per-point detection noise and is
// more stable when part of the iris is occluded by an eyelid.
function irisCentroid(normalized, ringIndices) {
  const pts = ringIndices.map(i => normalized[i]);
  return [
    pts.reduce((s, p) => s + p[0], 0) / pts.length,
    pts.reduce((s, p) => s + p[1], 0) / pts.length,
    pts.reduce((s, p) => s + p[2], 0) / pts.length,
  ];
}

function calculateDeviation(landmarks) {
  const { R, s, t } = alignmentMatrix(landmarks);
  const normalized = applyAlignment(landmarks, R, s, t);

  const predRt = predictBarycentricPoint(
    bary_model_rt.data.map(it => it.w_value),
    bary_model_rt.data.map(it => it.selected),
    normalized
  );

  const predLt = predictBarycentricPoint(
    bary_model_lt.data.map(it => it.w_value),
    bary_model_lt.data.map(it => it.selected),
    normalized
  );

  // FIX 1: Use centroid of iris ring points instead of single landmark.
  // Right eye ring: 469 (med), 470 (sup), 471 (lat), 472 (inf)
  // Left eye ring:  474 (lat), 475 (sup), 476 (med), 477 (inf)
  const rawRtCenter = irisCentroid(normalized, [469, 470, 471, 472]);
  const rawLtCenter = irisCentroid(normalized, [474, 475, 476, 477]);

  // FIX 2 (temporal smoothing): Apply EMA to iris centers to reduce
  // frame-to-frame jitter. smoothedRtCenter/smoothedLtCenter persist
  // across frames as module-level state.
  smoothedRtCenter = smoothPoint(smoothedRtCenter, rawRtCenter, SMOOTHING_ALPHA);
  smoothedLtCenter = smoothPoint(smoothedLtCenter, rawLtCenter, SMOOTHING_ALPHA);

  // FIX 3: Compute iris diameter entirely in normalized space using vecNorm.
  // Previously this used raw[x] coordinates only (x-axis span), which:
  //   (a) mixed coordinate spaces with the normalized offset above, and
  //   (b) shrank with head tilt, inflating deviation measurements.
  // vecNorm gives the true 3D distance, which is rotation-invariant.
  const rtIrisDiameter = vecNorm(
    vecSub(normalized[MP_RT_IRIS_MED], normalized[MP_RT_IRIS_LAT])
  );
  const ltIrisDiameter = vecNorm(
    vecSub(normalized[MP_LT_IRIS_MED], normalized[MP_LT_IRIS_LAT])
  );
  const maxIrisDiameter = Math.max(rtIrisDiameter, ltIrisDiameter);


  // Offset = how far the smoothed iris center is from the predicted
  // neutral position, normalized by iris diameter.
  const yawRtOffset = (smoothedRtCenter[0] - predRt[0]) / maxIrisDiameter;
  const yawLtOffset = (smoothedLtCenter[0] - predLt[0]) / maxIrisDiameter;

  const yawRt = Math.atan(yawRtOffset) * 180 / Math.PI;
  const yawLt = Math.atan(yawLtOffset) * 180 / Math.PI;

  const valid = Math.abs(yawRt) < 20 || Math.abs(yawLt) < 20;

 return {
    deviationPD: valid ? degreesToPD(yawRt - yawLt) : NaN,
    predictedRt: reverseAlignment([predRt], R, s, t)[0],
    predictedLt: reverseAlignment([predLt], R, s, t)[0],
    smoothedRt: smoothedRtCenter,
    smoothedLt: smoothedLtCenter
  };
}

console.log("===== loaded strab measurement helper functions successfully =====")

// ===========================================================
// =========== INIT FACE LANDMARKER ==========================
// ===========================================================

async function createFaceLandmarker() {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    )
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      runningMode,
      numFaces: 1
    })
    demosSection.classList.remove("invisible")
  } catch (err) {
    console.error(err);
  }
}
await createFaceLandmarker()

console.log(faceLandmarker)
console.log("===== loaded faceLandmarker successfully =====")

// ===========================================================
// =========== BEGIN WEBCAM INIT =============================
// ===========================================================

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
}

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton")
  enableWebcamButton.addEventListener("click", enableCam)
} else {
  console.warn("getUserMedia() is not supported by your browser")
}

function getMean(numbers) {
  if (numbers.length === 0) return 0;
  const sum = numbers.reduce((acc, cur) => acc + cur, 0);
  return sum / numbers.length;
}

function updateDeviationDisplay() {
  console.log(qualityControlPassedDeviationResults)
  if (qualityControlPassedDeviationResults.length <= 4) {
    return console.log("too few measurements, not able to update display")
  }
  const n = qualityControlPassedDeviationResults.length
  const mean = getMean(qualityControlPassedDeviationResults)
  const variance = qualityControlPassedDeviationResults.reduce(
    (a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
  const sd = Math.sqrt(variance)
  document.getElementById("readout").innerHTML = (
    "Deviation: " + mean.toFixed(2) + " ± " + sd.toFixed(2)
  )
}

function drawLastFrame() {
  const canvas = document.getElementById('video_freeze_canvas');
  const context = canvas.getContext('2d');
  canvas.style.transform =
    currentFacingMode === "user" ? "scaleX(-1)" : "scaleX(1)";
  const radio = video.videoHeight / video.videoWidth
  video.style.width = videoWidth + "px"
  video.style.height = videoWidth * radio + "px"
  canvas.style.width = videoWidth + "px"
  canvas.style.height = videoWidth * radio + "px"
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  canvas.style.display = 'block';
}

function clearLastFrame() {
  const canvas = document.getElementById('video_freeze_canvas');
  canvas.style.display = 'none'
}

async function enableCam(event) {
  if (!faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.")
    return
  }

  if (webcamRunning === true) {
    webcamRunning = false
    enableWebcamButton.innerText = "ENABLE PREDICTIONS"
    updateDeviationDisplay()
    qualityControlPassedDeviationResults = []
    resetSmoothing() // clear EMA state when session ends
  } else {
    webcamRunning = true
    clearLastFrame()
    enableWebcamButton.innerText = "DISABLE PREDICTIONS"
  }

  if (currentStream && !webcamRunning) {
    drawLastFrame()
    currentStream.getTracks().forEach(track => track.stop());
    return;
  }

  await startCamera();
}

async function startCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
  }

  webcamRunning = true
  resetSmoothing() // clear EMA state on camera restart
  enableWebcamButton.innerText = "DISABLE PREDICTIONS"

  const constraints = {
    video: {
      facingMode: { ideal: currentFacingMode },
      width: { ideal: 1280 },
      height: { ideal: 720 }
    }
  };

  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    video.addEventListener("loadeddata", predictWebcam);
    video.style.transform =
      currentFacingMode === "user" ? "scaleX(-1)" : "scaleX(1)";
    checkBrowserZoomCapabilities()
    checkCameraZoomCapabilities()
  } catch (err) {
    console.error("Camera error:", err);
  }
}

function checkBrowserZoomCapabilities() {
  if (navigator.mediaDevices.getSupportedConstraints().zoom) {
    console.log("Browser supports zoom");
  } else {
    alert("The browser does not support zoom.");
  }
}

const zoomSlider = document.getElementById("zoomInput");
zoomSlider.addEventListener("change", async () => {
  let expectedZoom = document.getElementById("zoomInput").value;
  const constraints = { advanced: [{ zoom: expectedZoom }] };
  const videoTracks = currentStream.getVideoTracks();
  let track = videoTracks[0];
  track.applyConstraints(constraints);
  track.addEventListener("loadeddata", predictWebcam);
});

function checkCameraZoomCapabilities() {
  const videoTracks = currentStream.getVideoTracks();
  let track = videoTracks[0];
  let capabilities = track.getCapabilities();
  if ('zoom' in capabilities) {
    let min = capabilities["zoom"]["min"];
    let max = capabilities["zoom"]["max"];
    document.getElementById("zoomInput").setAttribute("min", min);
    document.getElementById("zoomInput").setAttribute("max", max);
    document.getElementById("zoomInput").value = 1;
  } else {
    alert("This camera does not support zoom");
  }
}

console.log("===== loaded webcam init code successfully =====")

// ===========================================================
// =========== CAMERA TOGGLE CODE ============================
// ===========================================================

const switchLightBtn = document.getElementById("toggleLight");
switchLightBtn.addEventListener("click", async () => {
  currentLightOnStatus = !currentLightOnStatus;
  const constraints = { advanced: [{ torch: currentLightOnStatus }] };
  const videoTracks = currentStream.getVideoTracks();
  let track = videoTracks[0];
  track.applyConstraints(constraints);
});

const toggleOverlayBtn = document.getElementById("toggleOverlay");
toggleOverlayBtn.addEventListener("click", async () => {
  const overlay_canvas = document.getElementById('output_canvas');
  currentOverlayVisibleStatus = !currentOverlayVisibleStatus;
  overlay_canvas.style.display = currentOverlayVisibleStatus ? 'block' : 'none';
  const constraints = { advanced: [{ torch: currentLightOnStatus }] };
  const videoTracks = currentStream.getVideoTracks();
  let track = videoTracks[0];
  track.applyConstraints(constraints);
});

console.log("===== loaded camera toggle code successfully =====")

// ===========================================================
// =========== DRAWING CODE ==================================
// ===========================================================

let lastVideoTime = -1
let results = undefined
const drawingUtils = new DrawingUtils(canvasCtx)

async function predictWebcam() {
  if (webcamRunning === false
    || video.videoHeight === 0
    || video.videoWidth === 0) {
    return
  }

  canvasElement.style.transform =
    currentFacingMode === "user" ? "scaleX(-1)" : "scaleX(1)";

  const radio = video.videoHeight / video.videoWidth
  video.style.width = videoWidth + "px"
  video.style.height = videoWidth * radio + "px"
  canvasElement.style.width = videoWidth + "px"
  canvasElement.style.height = videoWidth * radio + "px"
  canvasElement.width = video.videoWidth
  canvasElement.height = video.videoHeight

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO"
    await faceLandmarker.setOptions({ runningMode: runningMode })
  }

  let startTimeMs = performance.now()
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime
    results = faceLandmarker.detectForVideo(video, startTimeMs)
  }

  if (results.faceLandmarks && results.faceLandmarks[0]) {
    let deviationResults = calculateDeviation(results.faceLandmarks[0])

    if (!Number.isNaN(deviationResults.deviationPD)) {
      qualityControlPassedDeviationResults.push(
        deviationResults.deviationPD * PREDICTED_DEVIATION_SCALE_FACTOR)
      if (qualityControlPassedDeviationResults.length > MAX_NUM_DEVIATION_OBSERVATIONS) {
        qualityControlPassedDeviationResults.shift()
      }
      document.getElementById("measurementProgress").value = qualityControlPassedDeviationResults.length
      document.getElementById("measurementProgress").max = MAX_NUM_DEVIATION_OBSERVATIONS
    }

   // Yellow dots = smoothed detected iris centers
    if (deviationResults.smoothedRt && deviationResults.smoothedLt) {
  drawingUtils.drawLandmarks(
    [{x: deviationResults.smoothedRt[0], y: deviationResults.smoothedRt[1], z: deviationResults.smoothedRt[2]},
     {x: deviationResults.smoothedLt[0], y: deviationResults.smoothedLt[1], z: deviationResults.smoothedLt[2]}],
    { color: "#FFFF00", lineWidth: 1 }
  )
}

    // Red dots = predicted neutral positions from barycentric model
    drawingUtils.drawLandmarks(
      [{
        "x": deviationResults.predictedRt[0],
        "y": deviationResults.predictedRt[1],
        "z": deviationResults.predictedRt[2]
      }, {
        "x": deviationResults.predictedLt[0],
        "y": deviationResults.predictedLt[1],
        "z": deviationResults.predictedLt[2]
      }], { color: "#FF0000", lineWidth: 1 }
    )

    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#C0C0C070" })
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#C0C0C070" })
    }
  }

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam)
  }
}

function drawBlendShapes(el, blendShapes) {
  if (!blendShapes.length) return;
  let htmlMaker = ""
  blendShapes[0].categories.map(shape => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `
  })
  el.innerHTML = htmlMaker
}

console.log("===== loaded drawing annotation code successfully =====")
console.log("===== loaded all code successfully =====")
