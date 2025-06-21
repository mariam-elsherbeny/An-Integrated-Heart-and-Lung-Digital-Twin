import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

const scene = new THREE.Scene();
scene.background = new THREE.Color("#f5f6f8");

let lungMixer = null,
    lungAction = null,
    lungModel = null;

let mixer = null,
    currentModel = null,
    currentAction = null;

let isPlaying = true;

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 40);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.physicallyCorrectLights = true;

function appendRenderer() {
    const container = document.getElementById("container3D");
    if (container) {
        container.appendChild(renderer.domElement);

        const updateSize = () => {
            const width = container.clientWidth;
            const height = container.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };

        updateSize();
        window.addEventListener("resize", updateSize);
    } else {
        console.error("Error: #container3D not found!");
    }
}

if (document.readyState === "complete") {
    appendRenderer();
} else {
    document.addEventListener("DOMContentLoaded", appendRenderer);
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.25;
controls.screenSpacePanning = true;
controls.enableZoom = true;
controls.maxDistance = 200;
controls.minDistance = 10;
controls.enableRotate = true;
controls.maxPolarAngle = Math.PI;
controls.minPolarAngle = 0;

// Lighting Setup
const mainLight = new THREE.DirectionalLight(0xffffff, 6);
mainLight.position.set(500, 500, 500);
scene.add(mainLight);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(100, 100, 100);
scene.add(dirLight);

const pointLight = new THREE.PointLight(0xffaa00, 1, 100);
pointLight.position.set(10, 10, 10);
scene.add(pointLight);

const spotLight = new THREE.SpotLight(0xffffff);
spotLight.position.set(15, 40, 35);
spotLight.angle = Math.PI / 6;
scene.add(spotLight);

const loader = new GLTFLoader();

function disposeModel(model) {
    if (!model) return;
    model.traverse((node) => {
        if (node.isMesh) {
            node.geometry.dispose();
            if (Array.isArray(node.material)) {
                node.material.forEach((mat) => mat.dispose());
            } else {
                node.material.dispose();
            }
        }
    });
    scene.remove(model);
}


async function loadScaleFactors(imageName) {
    const response = await fetch('/static/data/heart_dimensions_normalized.csv');
    const csvText = await response.text();

    const rows = csvText.trim().split('\n');
    const headers = rows[0].split(',');

    for (let i = 1; i < rows.length; i++) {
    const row = rows[i].split(',');
    const rowData = {};
    headers.forEach((header, index) => {
        rowData[header.trim()] = row[index].trim();
    });

    if (rowData.image_name === imageName) {
        return {
            x: parseFloat(rowData.x),
            y: parseFloat(rowData.y),
            z: parseFloat(rowData.z),
            };
        }
    }

    // Default scale if not found
    return { x: 1, y: 1, z: 1 };
}

function centerAndScaleModel(model, scale = 1, position = { x: 0, y: 0, z: 0 }) {
    const box = new THREE.Box3().setFromObject(model);
    const center = new THREE.Vector3();
    box.getCenter(center);
    model.position.sub(center);

    if (scale !== 1) {
        model.scale.set(scale, scale, scale);
    }

    if (position.x !== 0 || position.y !== 0 || position.z !== 0) {
        model.position.set(position.x, position.y, position.z);
    }
}

function fixTextureEncoding(model) {
    model.traverse((node) => {
        if (node.isMesh && node.material) {
            let materials = Array.isArray(node.material) ? node.material : [node.material];
            materials.forEach(mat => {
                if (mat.map) mat.map.encoding = THREE.sRGBEncoding;
                if (mat.emissiveMap) mat.emissiveMap.encoding = THREE.sRGBEncoding;
                if (mat.map) mat.map.needsUpdate = true;
                if (mat.emissiveMap) mat.emissiveMap.needsUpdate = true;
            });
        }
    });
}

function setupModelAnimations(model, animations) {
    if (animations.length > 0) {
        const newMixer = new THREE.AnimationMixer(model);
        const newAction = newMixer.clipAction(animations[0]);
        if (isPlaying) newAction.play();
        console.log(`✨ Animation loaded for ${model.name || "model"}`);
        return { mixer: newMixer, action: newAction };
    }
    return { mixer: null, action: null };
}

export async function loadModel(modelPath, imageName) {
    if (!modelPath) {
        console.error("Model path is required");
        return;
    }

    if (currentModel) {
        disposeModel(currentModel);
        if (mixer) {
            mixer.stopAllAction();
            mixer = null;
            currentAction = null;
        }
    }

    const modelUrl = `/models/${modelPath}`;
    const scaleFactors = await loadScaleFactors(imageName);

    // Load model using a Promise so we can await it
    const gltf = await new Promise((resolve, reject) => {
        loader.load(
            modelUrl,
            resolve,
            undefined,
            reject
        );
    });

    currentModel = gltf.scene;
    currentModel.name = "heart";

    fixTextureEncoding(currentModel);
    centerAndScaleModel(currentModel);

    // Apply the CSV-based scale factors
    currentModel.scale.set(scaleFactors.x, scaleFactors.y, scaleFactors.z);
    console.log(`✨ Heart model scaled to (${scaleFactors.x}, ${scaleFactors.y}, ${scaleFactors.z})`);

    scene.add(currentModel);

    const box = new THREE.Box3().setFromObject(currentModel);
    const center = new THREE.Vector3();
    box.getCenter(center);
    controls.target.copy(center);
    controls.update();

    const { mixer: newMixer, action: newAction } = setupModelAnimations(currentModel, gltf.animations);
    mixer = newMixer;
    currentAction = newAction;
}

export function loadLungsModel(modelPath = "normal-lung.glb") {
    if (!modelPath) {
        console.error("Lung model path is required");
        return;
    }

    if (lungModel) disposeModel(lungModel);
    if (lungMixer) {
        lungMixer.stopAllAction();
        lungMixer = null;
    }

    const modelUrl = `/models/${modelPath}`;

    loader.load(
        modelUrl,
        (gltf) => {
            lungModel = gltf.scene;
            lungModel.name = 'lungs';

            fixTextureEncoding(lungModel);

            const box = new THREE.Box3().setFromObject(lungModel);
            const center = new THREE.Vector3();
            box.getCenter(center);
            lungModel.position.sub(center);

            lungModel.scale.set(0.11, 0.11, 0.11);
            lungModel.position.set(0, 0, 0);
            lungModel.rotation.set(Math.PI, Math.PI, Math.PI);

            if (modelPath.toLowerCase().includes('pneumonia')) {
                lungModel.position.x = -2;
                lungModel.position.y = 47;
                lungModel.position.z = -15;
            } else {
                lungModel.position.y = 21;
                lungModel.position.z = -15;
            }

            scene.add(lungModel);
            const lungLight = new THREE.SpotLight(0xffeeee, 5);
            lungLight.angle = Math.PI / 5;
            lungLight.penumbra = 0.4;
            lungLight.decay = 2;
            lungLight.distance = 100;

            const lungCenter = new THREE.Vector3();
            new THREE.Box3().setFromObject(lungModel).getCenter(lungCenter);
            lungLight.position.set(lungCenter.x + 20, lungCenter.y + 20, lungCenter.z + 20);
            lungLight.target.position.copy(lungCenter);

            scene.add(lungLight);
            scene.add(lungLight.target);

            const { mixer: newLungMixer, action: newLungAction } = setupModelAnimations(
                lungModel,
                gltf.animations,
                lungMixer,
                lungAction
            );
            lungMixer = newLungMixer;
            lungAction = newLungAction;
        },
        undefined,
        (error) => {
            console.error("Error loading lungs model:", error);
        }
    );
}

let patientStatic = null;
    const params = new URLSearchParams(window.location.search);
    const doctorId = params.get("doctor_id");
    const patientId = params.get("patient_id");
    const mri = params.get("mri")

    document.getElementById("todashboard").addEventListener("click", () => {
        if (doctorId) {
        window.location.href = `/dashboard?doctor_id=${doctorId}${patientId ? `&patient_id=${patientId}` : ""}`;
        } else {
        window.location.href = `/simulation?doctor_id=${doctorId}${patientId ? `&patient_id=${patientId}` : ""}`;
        }
    });

    async function fetchPatientData(patientId) {
        try {
            const response = await fetch(`/api/patient/${patientId}`);
            patientStatic = await response.json();
            if (patientStatic.error) {
            console.error(patientStatic.error);
            return;
            }
            console.log("Fetched Patient Data.");
        } catch (error) {
            console.error("Error fetching patient data.", error);
        }
        }

    setInterval(async () => {
        if (patientStatic) {
            let simulated;
    
            if (patientId == 2) {
                // ✅ NORMAL PATIENT
                simulated = {
                    BP: Math.floor(Math.random() * 10 + 110),      // 110–119 mmHg
                    BP_dia: Math.floor(Math.random() * 10 + 70),       // 70–79 mmHg
                    PR: Math.floor(Math.random() * 41 + 60),           // 60–100 bpm
                    RR: Math.floor(Math.random() * 7 + 12),            // 12–18 breaths/min
                    Temp: (Math.random() * 0.5 + 36.5).toFixed(1),     // 36.5–37.0 °C
                    SpO2: Math.floor(Math.random() * 5 + 95),          // 95–100%
                    VT: Math.floor(Math.random() * 51 + 475)           // 475–525 mL centered around 500
                };
    
            } else if (patientId == 1) {
                // 🚨 HEART FAILURE PATIENT
                const crisis = Math.random() < 0.5;
    
                simulated = {
                    BP: crisis
                        ? Math.floor(Math.random() * 10 + 181)         // 181–190 mmHg
                        : Math.floor(Math.random() * 30 + 140),        // 140–169 mmHg
                    BP_dia: crisis
                        ? Math.floor(Math.random() * 5 + 121)          // 121–125 mmHg
                        : Math.floor(Math.random() * 10 + 90),         // 90–99 mmHg
                    PR: Math.floor(Math.random() * 11 + 100),          // 100–110 bpm
                    RR: Math.floor(Math.random() * 6 + 24),            // 24–29 breaths/min
                    Temp: (Math.random() * 1 + 36).toFixed(1),         // 36.0–37.0 °C
                    SpO2: Math.floor(Math.random() * 6 + 85),          // 85–90%
                    VT: Math.floor(Math.random() * 51 + 350)           // 350–400 mL (reduced due to HF)
                };
            }
    
            console.log("Live Generated Vitals:", simulated);

    // Combine static + simulated
    const row = {
        // Static
        AGE: patientStatic.age,
        SMOKING: patientStatic.smoking,
        ALCOHOL: patientStatic.alcohol,
        PRIOR_CMP: patientStatic.prior_cmp,
        CKD: patientStatic.ckd,
        GLUCOSE: patientStatic.glucose,
        UREA: patientStatic.urea,
        CREATININE: patientStatic.creatinine,
        BNP: patientStatic.bnp,
        RAISED_CARDIAC_ENZYMES: patientStatic.raised_cardiac_enzymes,
        ACS: patientStatic.acs,
        STEMI: patientStatic.stemi,
        HEART_FAILURE: patientStatic.heart_failure,
        HFREF: patientStatic.hfref,
        HFNEF: patientStatic.hfnef,
        VALVULAR: patientStatic.valvular,
        CHB: patientStatic.chb,
        SSS: patientStatic.sss,
        AKI: patientStatic.aki,
        AF: patientStatic.af,
        VT: patientStatic.vt,
        CARDIOGENIC_SHOCK: patientStatic.cardiogenic_shock,
        PULMONARY_EMBOLISM: patientStatic.pulmonary_embolism,
        CHEST_INFECTION : patientStatic.chest_infection,
        Weight: patientStatic.weight,
        Sex: patientStatic.sex,
        BMI: patientStatic.bmi,
        DM_y: patientStatic.dm_y,
        HTN_y: patientStatic.htn_y,
        Obesity: patientStatic.obesity,
        DLP: patientStatic.dlp,
        functionClass: patientStatic.function_class,
        FBS: patientStatic.fbs,
        CR: patientStatic.cr,
        TG: patientStatic.tg,
        LDL: patientStatic.ldl,
        HDL: patientStatic.hdl,
        BUN: patientStatic.bun,
        HB_y: patientStatic.hb_y,
        VHD: patientStatic.vhd,
        EF: patientStatic.EF,
        Dyspnea: patientStatic.dyspnea,
        Edema: patientStatic.edema,
        Hemoglobin: patientStatic.hemoglobin,
        Na: patientStatic.Na,
        K: patientStatic.K,

        // Dynamic
        BP : simulated.BP ,
        PR: simulated.PR,
        BP_dia: simulated.BP_dia,
        RR : simulated.RR,
        Temp: simulated.Temp,
        SpO2:simulated.SpO2,
        VT:simulated.VT
    };

    console.log("Combined row for ML Model.");
    console.log(row)

    try {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(row)
    });

    const result = await response.json();

    const condition_to_model = {
        1: ["Normal", "Normalmotion.glb"],
        2: ["Abnormal", "Abnormal.glb"],
        0: ["Heart Failure", "HeartFailure.glb"],
        };

    console.log("Predicted class for the new patient =", result.predicted);
    let predictedHeartModel = condition_to_model[result.predicted][1];
    console.log("Heart Model updated to :", predictedHeartModel);
    loadModel(predictedHeartModel, mri);
    let predictedHeartCondition = condition_to_model[result.predicted][0];

    const lungModelToLoad = determineLungModel(predictedHeartCondition, patientStatic, simulated);
    console.log("Selected Lung Model:", lungModelToLoad === "normal-lung.glb" ? "Normal" : lungModelToLoad === "pneumonia.glb" ? "Pneumonia" : lungModelToLoad);
    loadLungsModel(lungModelToLoad);

    // Update floating panel
    updateFloatingPanel(simulated, predictedHeartCondition, lungModelToLoad);

    } catch (error) {
        console.error("Error during prediction.", error);
    }

    }
}, 5000);


        if (patientId) {
            fetchPatientData(patientId);
        }


const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    if (mixer && isPlaying) mixer.update(delta);
    if (lungMixer && isPlaying) lungMixer.update(delta);
    controls.update();
    renderer.render(scene, camera);
}
animate();

function determineLungModel(predictedHeartCondition, patientStatic, simulated) {
    // Helper functions for "normal" ranges
    function isNormal(val, min, max) {
        return val >= min && val <= max;
    }
    function isNormalString(val) {
        return val === "N" || val === "Normal" || val === 0;
    }
    function isAbnormalString(val) {
        return val === "Y" || val === "Abnormal" || val === 1;
    }

    // Extract relevant values
    const heartFailure = patientStatic.heart_failure === 1 || patientStatic.heart_failure === "Y";
    const cad = patientStatic.acs === 1 || patientStatic.acs === "Y" || patientStatic.stemi === 1 || patientStatic.stemi === "Y";
    const bnp = patientStatic.bnp;
    const ef = simulated.EF;
    const raisedEnzymes = patientStatic.raised_cardiac_enzymes === 1 || patientStatic.raised_cardiac_enzymes === "Y";
    const chestInfection = patientStatic.chest_infection === 1 || patientStatic.chest_infection === "Y";
    const pulmonaryEmbolism = patientStatic.pulmonary_embolism === 1 || patientStatic.pulmonary_embolism === "Y";
    const functionClass = patientStatic.function_class || simulated.functionClass;
    const edema = simulated.edema || patientStatic.edema;
    const creatinine = patientStatic.creatinine;
    const urea = patientStatic.urea;
    const aki = patientStatic.aki === 1 || patientStatic.aki === "Y";

    // Scenario logic
    // 1. Normal Heart → Normal Lung
    if (!heartFailure && !cad && !chestInfection && !pulmonaryEmbolism &&
        isNormal(bnp, 0, 100) && isNormal(ef, 50, 60) && !raisedEnzymes &&
        isNormalString(functionClass) && !edema && isNormal(creatinine, 0.6, 1.3) && isNormal(urea, 7, 20)) {
        return "normal-lung.glb";
    }
    // 2. Normal Heart → Pneumonia Lung
    if (!heartFailure && !cad && (chestInfection || pulmonaryEmbolism ||
        isAbnormalString(functionClass) || edema || aki || creatinine > 1.3 || urea > 20)) {
        return "pneumonia.glb";
    }
    // 3. CAD Heart → Normal Lung
    if (cad && !heartFailure && !chestInfection && !pulmonaryEmbolism && !edema) {
        return "normal-lung.glb";
    }
    // 4. CAD Heart → Pneumonia Lung
    if (cad && (chestInfection || pulmonaryEmbolism || edema || isAbnormalString(functionClass) || urea > 20 || creatinine > 1.3)) {
        return "pneumonia.glb";
    }
    // 5. Heart Failure → Normal Lung
    if (heartFailure && bnp > 100 && ef < 50 && raisedEnzymes && !chestInfection && !pulmonaryEmbolism &&
        isNormalString(functionClass) && !edema && isNormal(creatinine, 0.6, 1.3) && isNormal(urea, 7, 20)) {
        return "normal-lung.glb";
    }
    // 6. Heart Failure → Pneumonia Lung
    if (heartFailure && (chestInfection || pulmonaryEmbolism || isAbnormalString(functionClass) || edema || creatinine > 1.3 || urea > 20 || aki)) {
        return "pneumonia.glb";
    }
    // Default
    return "normal-lung.glb";
}

// --- Floating Panel Update Logic ---
function updateFloatingPanel(simulated, predictedHeartCondition, lungModelToLoad) {
    const paramsList = document.getElementById('params-list');
    if (!paramsList) return;
    // Build parameter rows
    const paramRows = [
        { label: 'Pulse Rate (PR)', value: simulated.PR, unit: 'bpm' },
        { label: 'Diastolic BP (BP_dia)', value: simulated.BP_dia, unit: 'mmHg' },
        { label: 'Systolic BP (BP_sys)', value: simulated.BP, unit: 'mmHg' },
        { label: 'Respiratory Rate (RR)', value: simulated.RR, unit: 'breaths/min' },
        { label: 'Temperature (Temp)', value: simulated.Temp, unit: '°C' },
        { label: 'Oxygen Saturation (SpO₂)', value: simulated.SpO2, unit: '%' },
        { label: 'Tidal Volume (VT)', value: simulated.VT, unit: 'mL' }
    ];    
    paramsList.innerHTML = paramRows.map(row =>
        `<div class="flex justify-between py-0.5"><span>${row.label}</span><span class="font-semibold">${row.value}${row.unit ? ' ' + row.unit : ''}</span></div>`
    ).join('');

    // Color coding for heart
    const heartCond = document.getElementById('heart-condition');
    if (heartCond) {
        let color = 'text-gray-500';
        if (predictedHeartCondition === 'Normal') color = 'text-green-600';
        else if (predictedHeartCondition === 'Abnormal') color = 'text-yellow-500';
        else if (predictedHeartCondition === 'Heart Failure') color = 'text-red-600';
        heartCond.textContent = predictedHeartCondition;
        heartCond.className = `font-bold ${color}`;
    }
    // Color coding for lung
    const lungCond = document.getElementById('lung-condition');
    if (lungCond) {
        let lungText = 'Normal', color = 'text-green-600';
        if (lungModelToLoad === 'pneumonia.glb') { lungText = 'Pneumonia'; color = 'text-red-600'; }
        lungCond.textContent = lungText;
        lungCond.className = `font-bold ${color}`;
    }
}


