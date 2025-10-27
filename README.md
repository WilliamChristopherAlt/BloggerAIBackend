SETUP:

SET UP C++
    Vào https://visualstudio.microsoft.com/visual-cpp-build-tools/
    -> Download Build Tools -> mở file .exe tải về
    -> Workloads -> Tick Desktop development with C++ -> Giữ các lựa chọn tải về mặc định -> Install

SET UP REPO
    git clone https://github.com/WilliamChristopherAlt/BloggerAIBackend.git
    cd BloggerAIBackend
    python -m venv venv
    venv\Scripts\activate  # Windows
    pip install -r requirements.txt
  
SET UP MÔI TRƯỜNG LÀM VIỆC
    Tải về hai model:
        https://huggingface.co/facebook/bart-large-mnli/tree/main
              Vào folder models/bart-large-mnli -> tải file model.safetensors trong link
        https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
              Vào folder models/Mistral-7B-Instruct-v0.2-GGUF -> tải mistral-7b-instruct-v0.2.Q4_K_M.gguf trong link

CHẠY MODEL QUA API
    uvicorn fastapi_news_api:app --reload --host 127.0.0.1 --port 8000


SAU ĐÓ BUILD PROJECT BÊN MVC NHƯ THƯỜNG
