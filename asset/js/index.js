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
let lipsyncMain,
baselineObj,
model

function loadBaseline(){
    return new Promise((resolve) => {
        $.getJSON("/asset/baseline/data.json", (json) => {
            resolve(json)
        });
    });
}

function gameProcessing(result){
    document.getElementById("debug-score").innerHTML = Math.round(result.matchScore)
}

$(window).on('load', async function() {
    model = await facemesh.load();
    baselineObj = await loadBaseline()
    lipsyncMain = lipsync()
    await lipsyncMain.init('camera-video', gameProcessing, model)
    await lipsyncMain.setup(baselineObj)
    lipsyncMain.start()
});