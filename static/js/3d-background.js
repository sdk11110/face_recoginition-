/**
 * 3D科技感背景
 * 基于Three.js实现的动态网格和粒子系统
 */

class TechBackground {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id '${containerId}' not found`);
            return;
        }

        // 检测GPU性能并自动调整
        this.detectGPUCapability();

        // 默认配置
        this.options = Object.assign({
            particleCount: 1000,
            particleColor: 0x4a6fff,
            lineColor: 0x4a6fff,
            backgroundColor: 0x000823,
            gridColor: 0x1a2b4a,
            cameraDistance: 100,
            particleSize: 0.8,
            particleSpeed: 0.5,
            lineOpacity: 0.2,
            gridSize: 100,
            gridDivisions: 20,
            usePostProcessing: this.gpuPerformance === 'high', // 是否使用后处理效果
            autoAdjust: true // 是否自动调整性能
        }, options);

        // 如果启用了自动调整，根据GPU性能调整选项
        if (this.options.autoAdjust) {
            this.adjustOptionsForPerformance();
        }

        this.init();
    }

    // 检测GPU性能
    detectGPUCapability() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (!gl) {
                this.gpuPerformance = 'low';
                return;
            }
            
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (!debugInfo) {
                this.gpuPerformance = 'medium';
                return;
            }
            
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            
            // 检查是否有独立GPU
            const hasGPU = /(nvidia|amd|radeon|geforce|intel iris|intel hd)/i.test(renderer);
            
            // 估算性能等级
            if (hasGPU && /(rtx|gtx|radeon rx|quadro|vega)/i.test(renderer)) {
                this.gpuPerformance = 'high';
            } else if (hasGPU) {
                this.gpuPerformance = 'medium';
            } else {
                this.gpuPerformance = 'low';
            }
            
            console.log(`检测到GPU: ${renderer}, 性能评级: ${this.gpuPerformance}`);
            
        } catch (e) {
            console.error('GPU检测失败:', e);
            this.gpuPerformance = 'low'; // 默认为低性能
        }
    }
    
    // 根据性能调整选项
    adjustOptionsForPerformance() {
        switch (this.gpuPerformance) {
            case 'low':
                this.options.particleCount = Math.min(this.options.particleCount, 300);
                this.options.particleSize *= 0.8;
                this.options.particleSpeed *= 0.7;
                this.options.lineOpacity *= 0.5;
                this.options.gridDivisions = Math.min(this.options.gridDivisions, 10);
                break;
                
            case 'medium':
                this.options.particleCount = Math.min(this.options.particleCount, 600);
                this.options.particleSize *= 0.9;
                this.options.particleSpeed *= 0.85;
                this.options.lineOpacity *= 0.8;
                this.options.gridDivisions = Math.min(this.options.gridDivisions, 16);
                break;
                
            case 'high':
                // 高性能GPU保持原设置不变或稍微提升
                break;
        }
    }

    init() {
        // 创建场景
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        this.scene.fog = new THREE.FogExp2(this.options.backgroundColor, 0.001);

        // 创建相机
        const { width, height } = this.container.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.z = this.options.cameraDistance;

        // 创建渲染器
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: this.gpuPerformance !== 'low', // 低性能设备禁用抗锯齿
            alpha: true 
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(
            this.gpuPerformance === 'high' ? window.devicePixelRatio : 
            this.gpuPerformance === 'medium' ? Math.min(window.devicePixelRatio, 1.5) : 1
        );
        this.container.appendChild(this.renderer.domElement);

        // 添加网格
        this.addGrid();

        // 添加粒子系统
        this.addParticles();

        // 添加事件监听
        window.addEventListener('resize', this.onWindowResize.bind(this));
        document.addEventListener('mousemove', this.onDocumentMouseMove.bind(this), false);

        // 性能监控
        this.setupPerformanceMonitoring();

        // 开始动画
        this.animate();
    }

    setupPerformanceMonitoring() {
        // 帧率监测
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.averageFPS = 0;
        
        // 自动性能调整
        this.performanceAdjustmentCount = 0;
    }

    monitorPerformance() {
        this.frameCount++;
        const currentTime = performance.now();
        const elapsed = currentTime - this.lastTime;
        
        // 每秒更新一次FPS
        if (elapsed >= 1000) {
            const currentFPS = this.frameCount * 1000 / elapsed;
            this.averageFPS = this.averageFPS ? 
                (this.averageFPS * 0.8 + currentFPS * 0.2) : currentFPS;
            
            this.frameCount = 0;
            this.lastTime = currentTime;
            
            // 性能自动调整逻辑
            this.autoAdjustPerformance();
        }
    }
    
    autoAdjustPerformance() {
        // 限制调整频率，每10次检测才调整一次
        this.performanceAdjustmentCount++;
        if (this.performanceAdjustmentCount < 10) return;
        this.performanceAdjustmentCount = 0;
        
        // 如果帧率过低，减少粒子数量和特效
        if (this.averageFPS < 30 && this.options.particleCount > 100) {
            // 降低粒子数量20%
            const reduction = Math.floor(this.options.particleCount * 0.2);
            this.options.particleCount -= reduction;
            
            // 重新创建粒子系统
            this.scene.remove(this.particles);
            for (let line of this.lines) {
                this.scene.remove(line);
            }
            this.lines = [];
            
            this.addParticles();
            console.log(`性能自动调整: 降低粒子数量至 ${this.options.particleCount}`);
        }
    }

    addGrid() {
        // 添加网格
        const gridHelper = new THREE.GridHelper(
            this.options.gridSize, 
            this.options.gridDivisions, 
            this.options.gridColor, 
            this.options.gridColor
        );
        gridHelper.position.y = -20;
        gridHelper.material.opacity = 0.2;
        gridHelper.material.transparent = true;
        this.scene.add(gridHelper);
        this.grid = gridHelper;
    }

    addParticles() {
        // 创建粒子几何体
        const particlesGeometry = new THREE.BufferGeometry();
        const particlePositions = new Float32Array(this.options.particleCount * 3);
        const particleVelocities = [];

        for (let i = 0; i < this.options.particleCount; i++) {
            const x = (Math.random() - 0.5) * this.options.gridSize;
            const y = (Math.random() - 0.5) * this.options.gridSize;
            const z = (Math.random() - 0.5) * this.options.gridSize;

            particlePositions[i * 3] = x;
            particlePositions[i * 3 + 1] = y;
            particlePositions[i * 3 + 2] = z;

            // 速度向量
            particleVelocities.push({
                x: (Math.random() - 0.5) * this.options.particleSpeed,
                y: (Math.random() - 0.5) * this.options.particleSpeed,
                z: (Math.random() - 0.5) * this.options.particleSpeed
            });
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

        // 创建粒子材质
        const particlesMaterial = new THREE.PointsMaterial({
            color: this.options.particleColor,
            size: this.options.particleSize,
            transparent: true,
            blending: THREE.AdditiveBlending,
            opacity: 0.7,
            sizeAttenuation: true
        });

        // 创建粒子系统
        this.particles = new THREE.Points(particlesGeometry, particlesMaterial);
        this.scene.add(this.particles);
        this.particleVelocities = particleVelocities;

        // 创建连线几何体
        this.lineMaterial = new THREE.LineBasicMaterial({
            color: this.options.lineColor,
            transparent: true,
            opacity: this.options.lineOpacity
        });
        
        this.lines = [];
        this.updateLines();
    }

    updateLines() {
        // 移除旧线条
        for (let line of this.lines) {
            this.scene.remove(line);
        }
        this.lines = [];

        // 获取粒子位置
        const positions = this.particles.geometry.attributes.position.array;
        const maxDistance = 20;
        const maxLines = 100; // 限制线条数量，提高性能

        let lineCount = 0;
        // 创建新线条
        for (let i = 0; i < this.options.particleCount; i++) {
            if (lineCount >= maxLines) break;

            const x1 = positions[i * 3];
            const y1 = positions[i * 3 + 1];
            const z1 = positions[i * 3 + 2];

            for (let j = i + 1; j < this.options.particleCount; j++) {
                if (lineCount >= maxLines) break;

                const x2 = positions[j * 3];
                const y2 = positions[j * 3 + 1];
                const z2 = positions[j * 3 + 2];

                // 计算距离
                const distance = Math.sqrt(
                    Math.pow(x2 - x1, 2) + 
                    Math.pow(y2 - y1, 2) + 
                    Math.pow(z2 - z1, 2)
                );

                // 如果粒子足够近，则连线
                if (distance < maxDistance) {
                    const lineGeometry = new THREE.BufferGeometry();
                    const linePositions = new Float32Array([x1, y1, z1, x2, y2, z2]);
                    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
                    
                    const line = new THREE.Line(lineGeometry, this.lineMaterial);
                    this.scene.add(line);
                    this.lines.push(line);
                    lineCount++;
                }
            }
        }
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        this.monitorPerformance();
        this.update();
        this.render();
    }

    update() {
        // 更新粒子位置
        const positions = this.particles.geometry.attributes.position.array;
        const gridSize = this.options.gridSize / 2;

        for (let i = 0; i < this.options.particleCount; i++) {
            positions[i * 3] += this.particleVelocities[i].x;
            positions[i * 3 + 1] += this.particleVelocities[i].y;
            positions[i * 3 + 2] += this.particleVelocities[i].z;

            // 检查边界
            if (positions[i * 3] < -gridSize || positions[i * 3] > gridSize) {
                this.particleVelocities[i].x *= -1;
            }
            if (positions[i * 3 + 1] < -gridSize || positions[i * 3 + 1] > gridSize) {
                this.particleVelocities[i].y *= -1;
            }
            if (positions[i * 3 + 2] < -gridSize || positions[i * 3 + 2] > gridSize) {
                this.particleVelocities[i].z *= -1;
            }
        }

        this.particles.geometry.attributes.position.needsUpdate = true;

        // 定期更新线条 (每60帧更新一次以提高性能)
        if (Math.random() < 0.05) {
            this.updateLines();
        }

        // 旋转网格
        if (this.grid) {
            this.grid.rotation.y += 0.002;
        }

        // 缓慢旋转场景
        this.scene.rotation.y += 0.0005;
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const { width, height } = this.container.getBoundingClientRect();
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    onDocumentMouseMove(event) {
        // 鼠标交互效果
        const mouseX = (event.clientX / window.innerWidth) * 2 - 1;
        const mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
        
        this.scene.rotation.y = mouseX * 0.1;
        this.scene.rotation.x = mouseY * 0.1;
    }
} 