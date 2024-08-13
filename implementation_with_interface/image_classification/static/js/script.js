const canvas = document.getElementById('canvas');
const result=document.getElementById("result");
        const ctx = canvas.getContext('2d');
        const captureButton = document.getElementById('capture');
        let imageData;

        // Create and configure video element
        const video = document.createElement('video');
        document.body.appendChild(video);

        async function initCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();
            } catch (err) {
                console.error('Error accessing camera: ', err);
            }
        }

        captureButton.addEventListener('click', async () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            imageData = canvas.toDataURL('image/png');
            await uploadImage(imageData);
        });
        
        async function uploadImage(imageData) {
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `image=${encodeURIComponent(imageData)}`
                });
                const data = await response.json();
                result.textContent="predicted result is: "+data.class;
            } catch (err) {
                console.error('Error:', err);
            }
        }

        // Initialize camera
        initCamera();