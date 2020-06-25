/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

'use strict';
function lipsync() {
    let gameStarted = false,
    gameStoped = false,
    gamePause = false,
    matchScore = 0,
    dimensionsScore = 0,
    currentLyricsNum = 0, 
    latestScore = 0,
    score = 0, 
    faceDetectionAvailability = false,
    gameUpdate = ()=>{};

    let faceDistance = {};
    let reqPredMouthCoord = [];
    let meshArr = [];

    const MAX_CONTINUOUS_CHECKS = 5;

    let frameCanvas, 
    frameCanvasCTX,
    model, 
    videoid, 
    videoWidth, 
    videoHeight, 
    videoRatio, 
    cropCanvas, 
    cropCTX, 
    cropCanvasWidth, 
    cropCanvasHeight,
    cropCanvasX, 
    cropCanvasY,
    cropWidth, 
    cropHeight,  
    cropRatio, 
    cropX, 
    cropY, 
    video;

    let ctx, canvas;
    let compareCanvas1,compareCanvas2,compareCTX1,compareCTX2
    let compareCanvasWidth = 100, compareCanvasHeight = 100;
    let prediction;

    var baselineData;
    var baselineTime;
    var baselineIndex = 0;

    var mouthPoints = [
        78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95
    ];
    var vocalDone = false;
    var instrumentalDone = false;
    var cameraLoadingDone = false;
    var currentPlayTime = 0;

    var vocalBufferSource, instrumentalBufferSource;
    
    const color = 'red';

    const LANDMARKS_COUNT = 468;

    function distance(point1, point2) {
        const delta = Math.sqrt(Math.pow((point1[0] - point2[0]), 2) + Math.pow((point1[1] - point2[1]), 2))
        return delta;
    }

    function GetFaceFromBaseline(time) {
        if (baselineData == null) {
            return null;
        }
        for (let i = baselineIndex; i < baselineData.length - 1; ++i) {
            if(baselineData[i + 1][0]){
                if (baselineData[i + 1][0] > time+0.2) {
                    baselineIndex = i;
                    return JSON.parse(baselineData[i][1]);
                }
            }
        }
    }

    let curY = 500,
    curY2 = 500,
    effTrigger = false,
    fullEffIn = false,
    fadingEffectIn = false,
    fullEffOut = true,
    fadingEffectOut = false,
    effectTriggerTimeout = null,
    avoidFirstFrame = 10,
    mouthStartEffectTrigger = false,
    mouthEndEffectTrigger = false,
    mouthEndEffectActivated = false,
    mouthStartEffectActivated = false

    function DrawPredictedFace(framePrediction) {
        const keypoints = framePrediction;
        for (let i = 0; i+2 < TRIANGULATION.length; i+=3) {
            const x1 = keypoints[TRIANGULATION[i]][0];
            const y1 = keypoints[TRIANGULATION[i]][1];
            const z1 = keypoints[TRIANGULATION[i]][2];
            // console.log(z1);
            const x2 = keypoints[TRIANGULATION[i+1]][0];
            const y2 = keypoints[TRIANGULATION[i+1]][1];
            const z2 = keypoints[TRIANGULATION[i+1]][2];

            const x3 = keypoints[TRIANGULATION[i+2]][0];
            const y3 = keypoints[TRIANGULATION[i+2]][1];
            const z3 = keypoints[TRIANGULATION[i+2]][2];
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x3, y3);
            ctx.closePath();
            let activeVal
            if((y1 - curY)<(curY2 - y1)) activeVal = (y1 - curY)
            else activeVal = (curY2 - y1)
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.stroke();
        }
        if(avoidFirstFrame > 0) avoidFirstFrame -= 1
    }

    function CreateBinImage(mouthPoints, compareCTX){
        /* create binary image of mouth shape */
        compareCTX.clearRect(0, 0, compareCTX.canvas.width, compareCTX.canvas.height);
        compareCTX.beginPath();
      
        let leftpoint = mouthPoints[0]
        let toppoint = mouthPoints[5]
        let rightpoint = mouthPoints[10]
        let bottompoint = mouthPoints[15]
        let scaleratio = compareCanvasWidth/(rightpoint[0]-leftpoint[0])
              
        mouthPoints.map((elt, num) => {
          if(num === 0) compareCTX.moveTo((elt[0] - leftpoint[0])*scaleratio, (elt[1] - toppoint[1])*scaleratio);
          else{
            compareCTX.lineTo((elt[0] - leftpoint[0])*scaleratio, (elt[1] - toppoint[1])*scaleratio);
          }
        })
        compareCTX.fill();
        // CV Moment
        //find min and max points
        var minX = Math.min.apply(Math, mouthPoints.map(function(elt) { return elt[0]}));
        var minY = Math.min.apply(Math, mouthPoints.map(function(elt) { return elt[1]}));

        // move points to zero, to int
        mouthPoints.map((elt, num)=>{
          var newX = parseInt((elt[0] - minX) * 5)
          var newY = parseInt((elt[1] - minY) * 5)
          mouthPoints[num] = [newX, newY]
          return;});


        // create background image
        var imgWidth = Math.max.apply(Math, mouthPoints.map(function(elt) { return elt[0]}));
        var imgHigh = Math.max.apply(Math, mouthPoints.map(function(elt) { return elt[1]}));


        let img = new cv.Mat.zeros(imgHigh+1,imgWidth+1,cv.CV_8UC1 );
        
        // array to Mat
        var flatten_mouthPoints = [].concat.apply([], mouthPoints);
        let mat_mouthPoints = cv.matFromArray(20,1, cv.CV_32SC2 , flatten_mouthPoints);  
        let matvec_mouthPoints = new cv.MatVector();
        matvec_mouthPoints.push_back (mat_mouthPoints);
        let color = new cv.Scalar (255);   
        
        // fill poly 
        cv.fillPoly(img, matvec_mouthPoints,color); 
        
        
        // flush cv mat stuff
        mat_mouthPoints.delete();
        matvec_mouthPoints.delete();
        
        return img;
    }

    function ScoreMouth(framePrediction, baselineFace) {
        const keypoints = framePrediction;

        let predMouthCoord = [];
        let baseMouthCoord = [];
        reqPredMouthCoord = []
        // get mouth points from prediction
        for(let i = 0; i < mouthPoints.length; i++){
            let idx = mouthPoints[i];
            let tempCoord = [keypoints[idx][0], keypoints[idx][1]]
            predMouthCoord.push(tempCoord);     
            reqPredMouthCoord.push(tempCoord)
            baseMouthCoord.push([baselineFace[i][0], baselineFace[i][1]]);     
        }
        getFaceRatationPoint(keypoints)

        let predBin = CreateBinImage(predMouthCoord, compareCTX1)||[[0]];
        let baseBin = CreateBinImage(baseMouthCoord, compareCTX2)||[[0]];

        let shapeMatchDistant = cv.matchShapes(predBin, baseBin, cv.CONTOURS_MATCH_I2, 0)
        let finalScore = shapeMatchDistant
        return finalScore;
    }
    
    function getFaceRatationPoint(keypoints){
        const rotatePointArr = [];
        const facePoints = [10, 152, 5, 123, 352, 122, 167 ];
        for(let i = 0; i < facePoints.length; i++){
            let idx = facePoints[i];
            let tempCoord = [keypoints[idx][0], keypoints[idx][1]];  
            rotatePointArr.push(tempCoord);
        }

        faceDistance.rotatePoint = rotatePointArr;
    }

    function TryStart() {
        if (vocalDone && instrumentalDone && cameraLoadingDone && !gameStarted) {
            gameStarted = true;
            
            score = 0;
            currentLyricsNum = 0;
            latestScore = 0;
            currentPlayTime = 0;
            
            vocalBufferSource.start(0, 0);  
            instrumentalBufferSource.start(0);
        }
    }

    function TryContinue() {
        if(gamePause){
            gamePause = false
        }
    }

    function TryPause() {
        if(!gamePause){
            gamePause = true
        }
    }

    function TryStop() {
        if (gameStoped) {
            video.srcObject.getTracks()[0].stop();
        }
    }

    function getPrediction(pred){
        if(pred.length > 0){
            faceDetectionAvailability = true
            return pred[0]['scaledMesh']
        } 
        else {
            faceDetectionAvailability = false
            return;
        }
    }

    let workerCropCanvasCTX,
    kickStartSuccess = false,
    workerBaselineFace;
    

    async function kickStartWorkerPrediction(resolve){
        if(!kickStartSuccess){
            if(model){
                workerCropCanvasCTX = cropCanvas.getContext('2d');
                const predictionTensor = await model.estimateFaces(workerCropCanvasCTX.getImageData(0, 0, cropCanvas.width, cropCanvas.height))
                prediction = getPrediction(predictionTensor)
                kickStartSuccess = true
            }
            setTimeout(()=>kickStartWorkerPrediction(resolve), 500)
        }
        else resolve()
    }

    async function workerPrediction(){
        return new Promise(async res=>{
            frameCanvasCTX.drawImage(video,cropX,cropY,cropWidth,cropHeight,cropCanvasX,cropCanvasY,cropWidth,cropHeight);
            const predictionTensor = await model.estimateFaces(frameCanvas)
            if(predictionTensor.length > 0){
                if(predictionTensor[0]['faceInViewConfidence']>0.5){
                    prediction = getPrediction(predictionTensor)
                    faceDetectionAvailability = true
                }
                else{
                faceDetectionAvailability = false
                }
            }
            else{
                faceDetectionAvailability = false
            }
            workerBaselineFace = GetFaceFromBaseline(currentPlayTime);
            res(prediction)
        })
    }

    function Init(camid, updateFunction, predictor, permissionCallback=()=>{}){
        model = predictor
        gameUpdate = updateFunction
        return new Promise(async (resolve, reject) => {
            videoid = camid;
            let cameraPromise = new Promise(async (camresolve, camreject) => {
                try{
                    video = document.getElementById(videoid);
                    const stream = await navigator.mediaDevices.getUserMedia({
                        'audio': false,
                        'video': {
                        facingMode: 'user'
                        },
                    });
                    video.srcObject = stream;
                    video.onloadedmetadata = () => {
                        document.getElementById('camera-cropped-canvas').remove();
                        cropCanvas = null;
                        videoWidth = video.videoWidth;
                        videoHeight = video.videoHeight;
                        videoRatio = videoWidth/videoHeight;
                        cropRatio = 1
                        if(videoRatio>1){
                            cropHeight = videoHeight;
                            cropWidth = cropHeight * cropRatio;
                            cropX = videoWidth/2 - cropWidth/2;
                            cropY = 0;
                        }
                        else{
                            cropWidth = videoWidth;
                            cropHeight = cropWidth/cropRatio;
                            cropY = videoHeight/2 - cropHeight/2
                            cropX = 0;
                        }
                        cropCanvasWidth = cropWidth*1.74
                        cropCanvasHeight = cropHeight*1.74
                        cropCanvasX = (cropCanvasWidth - cropWidth)/2
                        cropCanvasY = 0.55*(cropCanvasHeight - cropHeight)/2
                        cropCanvas = document.createElement('canvas');
                        cropCanvas.id = "camera-cropped-canvas"
                        cropCanvas.width = cropCanvasWidth;
                        cropCanvas.height = cropCanvasHeight;
    
                        frameCanvas = document.createElement("canvas")
                        frameCanvasCTX = frameCanvas.getContext('2d')
                        frameCanvas.width = cropCanvas.width;
                        frameCanvas.height = cropCanvas.height;
                        frameCanvasCTX.translate(cropCanvas.width, 0);
                        frameCanvasCTX.scale(-1, 1);
    
                        document.getElementById('baseline-video-wrapper').appendChild(cropCanvas);
                        permissionCallback(true)
                        camresolve(video);
                    };
                }
                catch(err){
                    permissionCallback(false)
                    camreject(err)
                }
            });

            Promise.all([cameraPromise])
            .then(async res => {
                video.play();
                
                cropCTX = cropCanvas.getContext('2d');
                cropCTX.drawImage(video,cropX,cropY,cropWidth,cropHeight,cropCanvasX,cropCanvasY,cropWidth,cropHeight);
                setTimeout(()=>{
                    if(!kickStartSuccess) browser_checker().setIsSlow(true)
                }, 15000)
                kickStartWorkerPrediction(()=>resolve(true))
            })
            .catch(err => {
                console.log(err)
                console.log('Promise Fail :'+err)
                reject('fail')
            })
        });
    }
    
    function setUp(bufferobj) {
        return new Promise(async (resolve) => {
            baselineData = bufferobj;
            baselineTime = baselineData[baselineData.length-1][0]
            canvas = document.getElementById('camera-cropped-canvas');
            ctx = canvas.getContext('2d');

            compareCanvas1 = document.createElement('canvas');
            compareCanvas1.width = compareCanvasWidth;
            compareCanvas1.height = compareCanvasHeight;
            compareCTX1 = compareCanvas1.getContext("2d");

            compareCanvas2 = document.createElement('canvas');
            compareCanvas2.width = compareCanvasWidth;
            compareCanvas2.height = compareCanvasHeight;
            compareCTX2 = compareCanvas2.getContext("2d");
            
            let user_mouth_container = document.getElementById('your-prediction')
            let baseline_mouth_container = document.getElementById('baseline-prediction')
            user_mouth_container.appendChild(compareCanvas1)
            baseline_mouth_container.appendChild(compareCanvas2)
            
            Update(()=>{
                resolve()
            });
        });
    }
    async function Update(resolve) {
        if(!gameStoped){
            let currentScore
            await workerPrediction(workerBaselineFace)
            currentPlayTime += 0.1
            if(currentPlayTime >= baselineTime||!workerBaselineFace){
                currentPlayTime = 0.0
                baselineIndex = 0
                await workerPrediction(workerBaselineFace)
            }
            cropCTX.drawImage(frameCanvas,0,0);
            if(prediction){
                const baselineFace = workerBaselineFace;
                if(faceDetectionAvailability) DrawPredictedFace(prediction);
                
                if(prediction&&baselineFace) {
                    let baseVerticalMouthDistant = distance(baselineFace[5], baselineFace[15])
                    let baseHorizontalMouthDistant = distance(baselineFace[0], baselineFace[10])
                    let baseMouthRatio = Math.round(1000*baseVerticalMouthDistant/baseHorizontalMouthDistant)
                    let baseMouthActive = baseMouthRatio>50?true:false
                    if(baseMouthActive){
                        matchScore = ScoreMouth(prediction, baselineFace)
                        dimensionsScore = Math.max(0, (1.0 - matchScore));
                        dimensionsScore = 1 / (1 + Math.exp(-15*(2*dimensionsScore-1.3)))
                        currentScore = dimensionsScore*1000
                    }
                    else{
                        currentScore = 0
                    }
                }
            }

            if(!cameraLoadingDone){
                cameraLoadingDone = true;
                resolve()
            }

            let returnResult = {
                "score": score,
                "currentLyricsNum": currentLyricsNum,
                "latestScore": latestScore ,
                "songTime": currentPlayTime,
                "songStop": gameStoped,
                "faceDetected" : faceDetectionAvailability,
                "matchScore": currentScore
            }
            gameUpdate(returnResult);
            requestAnimationFrame(Update);
        }
    }

    async function start(){
        TryStart();
    }

    async function stop(){
        gameStoped = true;
        TryStop();
    }

    return {
        init: (id, updateFunction, predictor) => Init(id, updateFunction, predictor),
        setup: (bufferobj) => setUp(bufferobj),
        start: () => start(),
        pause: () => TryPause(),
        continue: () => TryContinue(),
        stop: () => stop(),
        toggleDebugMode: () => toggleDebugMode(),
    }
}