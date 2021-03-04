import m from 'mithril'


let appstate = {
    loading_webcam: true,
    detecting_faces: false,
    image_src: null
};

class Loader {
    view() {
        return [
            m("svg", {"class":"text-blue-500 animate-spin svg-inline--fa fa-cog fa-w-16 loader","aria-hidden":"true","focusable":"false","data-prefix":"fas","data-icon":"cog","role":"img","xmlns":"http://www.w3.org/2000/svg","viewBox":"0 0 512 512"}, 
              m("path", {"fill":"currentColor","d":"M487.4 315.7l-42.6-24.6c4.3-23.2 4.3-47 0-70.2l42.6-24.6c4.9-2.8 7.1-8.6 5.5-14-11.1-35.6-30-67.8-54.7-94.6-3.8-4.1-10-5.1-14.8-2.3L380.8 110c-17.9-15.4-38.5-27.3-60.8-35.1V25.8c0-5.6-3.9-10.5-9.4-11.7-36.7-8.2-74.3-7.8-109.2 0-5.5 1.2-9.4 6.1-9.4 11.7V75c-22.2 7.9-42.8 19.8-60.8 35.1L88.7 85.5c-4.9-2.8-11-1.9-14.8 2.3-24.7 26.7-43.6 58.9-54.7 94.6-1.7 5.4.6 11.2 5.5 14L67.3 221c-4.3 23.2-4.3 47 0 70.2l-42.6 24.6c-4.9 2.8-7.1 8.6-5.5 14 11.1 35.6 30 67.8 54.7 94.6 3.8 4.1 10 5.1 14.8 2.3l42.6-24.6c17.9 15.4 38.5 27.3 60.8 35.1v49.2c0 5.6 3.9 10.5 9.4 11.7 36.7 8.2 74.3 7.8 109.2 0 5.5-1.2 9.4-6.1 9.4-11.7v-49.2c22.2-7.9 42.8-19.8 60.8-35.1l42.6 24.6c4.9 2.8 11 1.9 14.8-2.3 24.7-26.7 43.6-58.9 54.7-94.6 1.5-5.5-.7-11.3-5.6-14.1zM256 336c-44.1 0-80-35.9-80-80s35.9-80 80-80 80 35.9 80 80-35.9 80-80 80z"})
            )
        ];
    }
}

class WebCam {
    oncreate(vnode) {
        var video = vnode.dom;

        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                appstate.loading_webcam = false;
                m.redraw();
                video.srcObject = stream;
            })
            .catch(function (e) {
                console.log("An error occurred while initializing webcam: " + e);
            });
        }
    }

    view(vnode) {

        return [
            m('video.video-container__element', {autoplay: 'true', id: 'videoElement', style: "display: " + (appstate.loading_webcam ? "none" : "block")})  
        ];
    }
}

class Home {
    take_picture() {
        appstate.detecting_faces = true;
        m.redraw();

        let video = document.getElementById('videoElement');
        let canvas = document.createElement('canvas');

        let width = video.offsetWidth;
        let height = video.offsetHeight;

        canvas.width = width;
        canvas.height = height;

        let context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, width, height);

        let data = canvas.toDataURL('image/png');
        // appstate.image_src = data;
        
        m.request({
            method: "POST",
            url: '/predict_image',
            responseType: "json",
            body: {
                image_data: data
            }
         })
        .then(d => {
            // console.log(d);
            context.lineWidth = 2;
            context.font = "12px Arial";


            d.forEach(face => {
                let conf = Math.round((face.conf*1) * 10000) / 100;
                


                if (face.label*1 == 1) {  
                    context.fillStyle = "#FF0000";
                    context.strokeStyle = "#FF0000";
                    
                    context.fillRect(face.x1*1 - 1, face.y1*1 - 20, 150, 20);

                    context.fillStyle = "#FFFFFF";
                    context.fillText("Unmasked face (" + conf + "%)", face.x1*1 + 5, face.y1*1 - 5);
                } else {
                    context.fillStyle = "#00FF00";
                    context.strokeStyle = "#00FF00";

                    context.fillRect(face.x1*1 - 1, face.y1*1 - 20, 150, 20);

                    context.fillStyle = "#000000";
                    context.fillText("Masked face (" + conf + "%)", face.x1*1 + 5, face.y1*1 - 5);
                }

                context.beginPath();
                context.rect(face.x1*1, face.y1*1, (face.x2*1 - face.x1*1), (face.y2*1 - face.y1*1));
                context.stroke();

            });

            let data = canvas.toDataURL('image/png');
            appstate.image_src = data;

            appstate.detecting_faces = false;

            m.redraw();
         })
        .catch(e => {
            alert("An error occurred while predicting image! Please refresh page and try again");
            console.log("Error while predicting image: "+ e);
            appstate.detecting_faces = false;
        })
    }

    oncreate() {
        setInterval(e => {
            if (!appstate.loading_webcam && !appstate.detecting_faces)
                this.take_picture();
        }, 50);
    }

    view() {
        return [
            m('.main-container', [
                m('.detection-container', [
                    m('.left-side', [
                        m('.video-container.shadow-lg.bg-white.rounded-lg', [
                            m(WebCam),

                            appstate.loading_webcam &&
                            m('.flex', [
                                m(Loader),
                                m('p.text-blue-500.ml-3.font-bold.text-xl', 'Loading camera...')
                            ]),

                            !appstate.loading_webcam &&
                            m('button.predict-btn.focus:outline-none.text-blue-600.text-sm.py-2.5.px-5.rounded-md.border.border-blue-600.hover:bg-blue-50.font-semibold.shadow-lg', {
                                onclick: e => this.take_picture()
                            }, 'Detect faces'),
                            // m('div', 'Loading camera...')
                        ]),

                    ]),

                    m('.right-side', [
                        // !appstate.loading_webcam &&
                        m('.image-container.shadow-lg.bg-white.rounded-lg', [
                            // m('img', {src: appstate.image_src != null ? appstate.image_src : })
                            (appstate.image_src != null) &&
                            m('div', m('img', { src: appstate.image_src })),
                            
                            // appstate.detecting_faces &&
                            // m('.flex', [
                            //     m(Loader),
                            //     m('p.text-blue-500.ml-3.font-bold.text-xl', 'Detecting faces...')
                            // ]),

                            (!appstate.detecting_faces && appstate.image_src == null) &&
                            m('p.text-blue-500.font-bold.text-xl', 'Image with detected faces will be shown here')
                        ])
                    ])
                ])
            ])
        ];
    }
}

export default Home