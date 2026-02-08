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

				this.addDOMWidget("3DViewer", "div", container);

                this.initViewer(container, cameraStateWidget);

				return r;
			};

            nodeType.prototype.initViewer = async function(container, cameraStateWidget) {
                await ensureThree();
                const THREE = window.THREE;

                // Use ResizeObserver to handle container size changes
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setPixelRatio(window.devicePixelRatio);
                container.appendChild(renderer.domElement);

                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x111111);

                const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
                camera.position.set(0, 0, 5);

                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                const pointsGroup = new THREE.Group();
                scene.add(pointsGroup);

                const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                scene.add(ambientLight);

                const animate = () => {
                    if (!this.preview_enabled) return;
                    requestAnimationFrame(animate);

                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    if (renderer.domElement.width !== width || renderer.domElement.height !== height) {
                        renderer.setSize(width, height, false);
                        camera.aspect = width / height;
                        camera.updateProjectionMatrix();
                    }

                    controls.update();
                    renderer.render(scene, camera);

                    // Update camera state widget
                    if (cameraStateWidget) {
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

                this.preview_enabled = true;
                animate();

                this.loadPLY = (url) => {
                    const loader = new THREE.PLYLoader();
                    loader.load(url, (geometry) => {
                        pointsGroup.clear();

                        // Check if geometry has colors
                        const material = new THREE.PointsMaterial({
                            size: 0.01,
                            vertexColors: geometry.attributes.color ? true : false
                        });

                        if (!geometry.attributes.color) {
                            material.color = new THREE.Color(0xffffff);
                        }

                        const points = new THREE.Points(geometry, material);
                        pointsGroup.add(points);

                        // Center camera
                        geometry.computeBoundingSphere();
                        const center = geometry.boundingSphere.center;
                        const radius = geometry.boundingSphere.radius;
                        controls.target.copy(center);
                        camera.position.set(center.x, center.y, center.z + radius * 2);
                        camera.near = radius / 100;
                        camera.far = radius * 100;
                        camera.updateProjectionMatrix();
                        controls.update();
                    },
                    (xhr) => { console.log((xhr.loaded / xhr.total * 100) + '% loaded'); },
                    (error) => { console.error('An error happened', error); });
                };

                this.onRemoved = () => {
                    this.preview_enabled = false;
                    renderer.dispose();
                };
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.ply_path) {
                    // In ComfyUI, files from 'output' or 'input' are served via /view
                    const url = api.api_url("/view?filename=" + encodeURIComponent(message.ply_path) + "&type=output");
                    this.loadPLY(url);
                }
            };
		}
	},
});
