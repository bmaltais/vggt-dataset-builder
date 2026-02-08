import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Load dependencies
async function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

let threeLoaded = false;
async function ensureThree() {
    if (threeLoaded) return;
    // NOTE: Loading from CDN for simplicity. For production use, consider:
    // - Bundling these dependencies with the extension
    // - Using subresource integrity (SRI) for security
    // - Providing offline/air-gapped support
    // Using a specific version of Three.js that is compatible with the examples
    await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js");
    await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js");
    await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js");
    threeLoaded = true;
}

app.registerExtension({
	name: "VGGT.Viewer",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "VGGT_PLY_Viewer") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const cameraStateWidget = this.widgets.find((w) => w.name === "camera_state");
				if (cameraStateWidget) {
                    cameraStateWidget.type = "hidden";
                }

				const container = document.createElement("div");
				container.style.width = "100%";
				container.style.height = "400px";
				container.style.position = "relative";
                container.style.backgroundColor = "black";
                container.style.borderRadius = "4px";
                container.style.marginTop = "10px";
                container.style.marginBottom = "10px";
                container.style.overflow = "hidden";

				this.addDOMWidget("3DViewer", "div", container);

                // Add a overlay for camera selection
                const overlay = document.createElement("div");
                overlay.style.position = "absolute";
                overlay.style.top = "10px";
                overlay.style.left = "10px";
                overlay.style.color = "white";
                overlay.style.background = "rgba(0,0,0,0.5)";
                overlay.style.padding = "5px";
                overlay.style.borderRadius = "4px";
                overlay.style.fontSize = "12px";
                overlay.style.pointerEvents = "none";
                container.appendChild(overlay);

                const cameraSelect = document.createElement("select");
                cameraSelect.style.pointerEvents = "auto";
                cameraSelect.style.marginTop = "5px";
                cameraSelect.style.display = "none";
                overlay.appendChild(document.createElement("div")).textContent = "Cameras:";
                overlay.appendChild(cameraSelect);

                // Store initialization promise
                this.viewerInitialized = this.initViewer(container, cameraStateWidget, cameraSelect);

				return r;
			};

            nodeType.prototype.initViewer = async function(container, cameraStateWidget, cameraSelect) {
                await ensureThree();
                const THREE = window.THREE;

                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x111111);

                const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
                camera.position.set(0, 0, 5);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                // Update camera state widget only when camera changes
                const updateCameraState = () => {
                    if (cameraStateWidget && !this.snapping) {
                        const viewMatrix = camera.matrixWorldInverse.toArray();
                        const projMatrix = camera.projectionMatrix.toArray();
                        const fov_y = camera.fov * Math.PI / 180;

                        const state = {
                            view_matrix: viewMatrix,
                            proj_matrix: projMatrix,
                            fov_y: fov_y
                        };
                        cameraStateWidget.value = JSON.stringify(state);
                    }
                };
                controls.addEventListener('change', updateCameraState);

                const pointsGroup = new THREE.Group();
                scene.add(pointsGroup);

                const camerasGroup = new THREE.Group();
                scene.add(camerasGroup);

                const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
                directionalLight.position.set(1, 1, 1);
                scene.add(directionalLight);

                const animate = () => {
                    if (!this.preview_enabled) return;
                    requestAnimationFrame(animate);

                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    const currentSize = renderer.getSize(new THREE.Vector2());
                    if (currentSize.x !== width || currentSize.y !== height) {
                        renderer.setSize(width, height, false);
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();
                    }

                    controls.update();
                    renderer.render(scene, camera);
                };

                this.preview_enabled = true;
                animate();

                this.loadPLY = (url) => {
                    const loader = new THREE.PLYLoader();
                    loader.load(url, (geometry) => {
                        // Dispose old geometries and materials before clearing
                        pointsGroup.children.forEach(child => {
                            if (child.geometry) child.geometry.dispose();
                            if (child.material) child.material.dispose();
                        });
                        pointsGroup.clear();
                        
                        const material = new THREE.PointsMaterial({
                            size: 0.01,
                            vertexColors: geometry.attributes.color ? true : false
                        });
                        if (!geometry.attributes.color) material.color = new THREE.Color(0xffffff);

                        const points = new THREE.Points(geometry, material);
                        pointsGroup.add(points);

                        geometry.computeBoundingSphere();
                        const center = geometry.boundingSphere.center;
                        const radius = geometry.boundingSphere.radius;
                        controls.target.copy(center);
                        if (!this.hasCameras) {
                            camera.position.set(center.x, center.y, center.z + radius * 2);
                            camera.lookAt(center);
                        }
                        camera.near = radius / 1000;
                        camera.far = radius * 1000;
                        camera.updateProjectionMatrix();
                        controls.update();
                    });
                };

                this.visualizeCameras = (cameras) => {
                    camerasGroup.clear();
                    this.hasCameras = cameras.length > 0;

                    cameraSelect.innerHTML = "";
                    const defaultOption = document.createElement("option");
                    defaultOption.textContent = "-- Select to Snap --";
                    cameraSelect.appendChild(defaultOption);

                    cameras.forEach((cam, index) => {
                        const viewMatrix = new THREE.Matrix4().fromArray(cam.view_matrix);
                        const projMatrix = new THREE.Matrix4().fromArray(cam.proj_matrix);

                        const helperCamera = new THREE.PerspectiveCamera();
                        helperCamera.projectionMatrix.copy(projMatrix);
                        // viewMatrix is worldToCamera, so we need cameraToWorld
                        helperCamera.matrixWorld.copy(viewMatrix.clone().invert());
                        helperCamera.matrixWorldInverse.copy(viewMatrix);

                        const helper = new THREE.CameraHelper(helperCamera);
                        helper.setColors(new THREE.Color(0xff0000), new THREE.Color(0x00ff00), new THREE.Color(0x0000ff), new THREE.Color(0xffff00), new THREE.Color(0xff00ff));
                        camerasGroup.add(helper);

                        const option = document.createElement("option");
                        option.value = index;
                        option.textContent = `Camera ${index}`;
                        cameraSelect.appendChild(option);
                    });

                    cameraSelect.style.display = cameras.length > 0 ? "block" : "none";

                    cameraSelect.onchange = () => {
                        const index = cameraSelect.value;
                        if (index !== "" && cameras[index]) {
                            const cam = cameras[index];
                            const viewMatrix = new THREE.Matrix4().fromArray(cam.view_matrix);
                            const cameraToWorld = viewMatrix.clone().invert();

                            this.snapping = true;

                            // Extract position and rotation from cameraToWorld
                            const position = new THREE.Vector3();
                            const quaternion = new THREE.Quaternion();
                            const scale = new THREE.Vector3();
                            cameraToWorld.decompose(position, quaternion, scale);

                            camera.position.copy(position);
                            camera.quaternion.copy(quaternion);
                            camera.fov = cam.fov_y * 180 / Math.PI;
                            camera.updateProjectionMatrix();

                            // Set control target to something in front of the camera
                            const target = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).add(position);
                            controls.target.copy(target);
                            controls.update();

                            this.snapping = false;

                            // Force widget update
                            if (cameraStateWidget) {
                                const state = {
                                    view_matrix: cam.view_matrix,
                                    proj_matrix: cam.proj_matrix,
                                    fov_y: cam.fov_y
                                };
                                cameraStateWidget.value = JSON.stringify(state);
                            }
                        }
                    };
                };

                this.onRemoved = () => {
                    this.preview_enabled = false;
                    // Dispose all geometries and materials
                    pointsGroup.children.forEach(child => {
                        if (child.geometry) child.geometry.dispose();
                        if (child.material) child.material.dispose();
                    });
                    camerasGroup.children.forEach(child => {
                        if (child.geometry) child.geometry.dispose();
                        if (child.material) child.material.dispose();
                    });
                    renderer.dispose();
                    if (container.contains(renderer.domElement)) {
                        container.removeChild(renderer.domElement);
                    }
                };
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);
                // Wait for viewer initialization before calling loadPLY
                if (this.viewerInitialized) {
                    await this.viewerInitialized;
                }
                if (message?.ply_path && this.loadPLY) {
                    const url = api.api_url("/view?filename=" + encodeURIComponent(message.ply_path) + "&type=output");
                    this.loadPLY(url);
                }
                if (message?.cameras && this.visualizeCameras) {
                    this.visualizeCameras(message.cameras);
                }
            };
		}
	},
});
