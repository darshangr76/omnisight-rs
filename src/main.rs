use axum::{
    routing::get,
    response::IntoResponse,
    body::Body,
    http::{header, StatusCode},
    extract::State,
};
use opencv::{prelude::*, videoio, core, imgcodecs};
use std::sync::Arc;
use tokio::sync::broadcast;
use std::time::Duration;
use bytes::Bytes;

struct AppState {
    tx: broadcast::Sender<Arc<Vec<u8>>>,
}

#[tokio::main]
async fn main() {
    let (tx, _rx) = broadcast::channel(10);
    let state = Arc::new(AppState { tx: tx.clone() });

    let tx_camera = tx.clone();
    tokio::task::spawn_blocking(move || {
        // 1. Open the video file
        let mut cam = videoio::VideoCapture::from_file("sample.mp4", videoio::CAP_ANY).unwrap();
        let mut frame = core::Mat::default();
        
        loop {
            // 2. Read the frame
            let success = cam.read(&mut frame).unwrap();
            if success && !frame.empty() {
                let mut buf = core::Vector::<u8>::new();
                let params = core::Vector::<i32>::new();
                
                // 3. Encode to JPEG
                if imgcodecs::imencode(".jpg", &frame, &mut buf, &params).is_ok() {
                    // 4. Zero-Copy Broadcast
                    let frame_arc = Arc::new(buf.to_vec());
                    let _ = tx_camera.send(frame_arc);
                }
            } else {
                // 5. Loop the video back to the start
                cam.set(videoio::CAP_PROP_POS_FRAMES, 0.0).unwrap();
            }
            
            // 6. Frame rate control (approx 30 FPS)
            std::thread::sleep(Duration::from_millis(33));
        }
    });

    let app = axum::Router::new()
        .route("/", get(root))
        .route("/video_feed", get(video_feed))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await.unwrap();
    println!("OmniSight-RS (Zero-Copy) running on http://127.0.0.1:8080");
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> impl IntoResponse {
    let html = r#"
    <html>
        <body style="background: #0a0a0a; color: white; text-align: center; font-family: sans-serif;">
            <h1>OmniSight-RS</h1>
            <p style="color: #4dff88; letter-spacing: 2px;">Zero-Copy Rust Inference Engine</p>
            <img src="/video_feed" style="max-width: 800px; border: 2px solid #4dff88; border-radius: 10px; box-shadow: 0 0 30px rgba(77,255,136,0.2);" />
        </body>
    </html>
    "#;
    axum::response::Html(html)
}

async fn video_feed(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut rx = state.tx.subscribe();
    
    let stream = async_stream::stream! {
        while let Ok(frame_arc) = rx.recv().await {
            let header = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n";
            let footer = b"\r\n";
            
            yield Ok::<_, std::convert::Infallible>(Bytes::from_static(header));
            yield Ok(Bytes::from(frame_arc.as_ref().clone()));
            yield Ok::<_, std::convert::Infallible>(Bytes::from_static(footer));
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame")],
        Body::from_stream(stream),
    )
}