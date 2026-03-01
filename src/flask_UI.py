from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, Response
import pyodbc
import hashlib
import os
import time
import json
import re
import secrets
from datetime import datetime
from dotenv import load_dotenv
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import random
import socket
import threading
import requests


# ============================================================
# Flask = UI + Auth + Session + DB
# Chat/RAG/LLM = FastAPI service (api_chat.py + rag_core.py)
# ============================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("Missing SECRET_KEY. Set SECRET_KEY in your environment or .env file.")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
if os.getenv("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True

# EMAIL CONFIG
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
mail = Mail(app)

FASTAPI_BASE_URL = (os.getenv("FASTAPI_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
gemini_ready = False


# ============================================================
# SQL Server
# ============================================================

def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=MSI;"
        "DATABASE=ChatbotDB;"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )


def create_usertable():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Users' AND xtype='U')
        CREATE TABLE Users (
            id INT IDENTITY(1,1) PRIMARY KEY,
            username NVARCHAR(100) UNIQUE NOT NULL,
            email NVARCHAR(255),
            password_hash NVARCHAR(512) NOT NULL,
            role NVARCHAR(50) DEFAULT 'user',
            created_at DATETIME DEFAULT GETDATE(),
            updated_at DATETIME DEFAULT GETDATE()
        )
        """
    )
    try:
        cursor.execute(
            """
            IF EXISTS (SELECT * FROM sysobjects WHERE name='Users' AND xtype='U')
            ALTER TABLE Users ALTER COLUMN password_hash NVARCHAR(512) NOT NULL
            """
        )
    except Exception as e:
        print(f"⚠️ Could not alter password_hash column size: {e}")
    conn.commit()
    conn.close()


def create_chattable():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Chats' AND xtype='U')
        CREATE TABLE Chats (
            id INT IDENTITY(1,1) PRIMARY KEY,
            username NVARCHAR(100),
            role NVARCHAR(10),
            message NVARCHAR(MAX),
            timestamp DATETIME DEFAULT GETDATE()
        )
        """
    )
    conn.commit()
    conn.close()


_init_lock = threading.Lock()
_initialized = False


def initialize_once():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        create_usertable()
        create_chattable()
        _initialized = True


# ============================================================
# CSRF
# ============================================================

CSRF_PROTECTED_ENDPOINTS = {
    "login",
    "register",
    "verify_email",
    "forgot_password",
    "verify_reset_otp",
    "reset_password_form",
}


def generate_csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = generate_csrf_token


@app.before_request
def bootstrap_app():
    if not _initialized:
        initialize_once()


@app.before_request
def enforce_csrf():
    if request.method == "POST" and request.endpoint in CSRF_PROTECTED_ENDPOINTS:
        token = request.form.get("csrf_token")
        if not token or token != session.get("csrf_token"):
            return "Invalid CSRF token", 400


# ============================================================
# User management
# ============================================================

def hash_password(password):
    return generate_password_hash(password)


def is_legacy_sha256_hash(value):
    return bool(re.fullmatch(r"[a-f0-9]{64}", value or ""))


def verify_password(stored_hash, password):
    if not stored_hash:
        return False
    if is_legacy_sha256_hash(stored_hash):
        return stored_hash == hashlib.sha256(password.encode()).hexdigest()
    return check_password_hash(stored_hash, password)


def upgrade_password_hash(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE Users SET password_hash=?, updated_at=GETDATE() WHERE username=?",
        (hash_password(password), username),
    )
    conn.commit()
    conn.close()


def add_userdata(username, email, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username=? OR email=?", (username, email))
    existing_user = cursor.fetchone()
    if existing_user:
        conn.close()
        return False

    if not email or not email.strip():
        conn.close()
        return False

    email_lower = email.lower()
    if not email_lower.endswith(".edu") and not email_lower.endswith(".edu.vn"):
        conn.close()
        return False

    cursor.execute(
        "INSERT INTO Users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email.strip(), hash_password(password)),
    )
    conn.commit()
    conn.close()
    return True


def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM Users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    stored_hash = row[0]
    if not verify_password(stored_hash, password):
        return False

    if is_legacy_sha256_hash(stored_hash):
        upgrade_password_hash(username, password)

    return True


def reset_password(username, new_password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE Users SET password_hash=?, updated_at=GETDATE() WHERE username=?",
        (hash_password(new_password), username),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


# ============================================================
# Chat history (DB)
# ============================================================

def save_chat(username, role, message):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO Chats (username, role, message) VALUES (?, ?, ?)",
        (username, role, message),
    )
    conn.commit()
    conn.close()


def get_chat_history(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM Chats WHERE username=? ORDER BY id ASC", (username,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


# ============================================================
# FastAPI helpers
# ============================================================

def fastapi_status():
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/status", timeout=5)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        return {"error": str(e)}


def fastapi_post(path, payload=None, timeout=30):
    url = f"{FASTAPI_BASE_URL}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.content else {}


# ============================================================
# Routes
# ============================================================

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("chatbot"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Check for admin logic
        admin_user = os.getenv("ADMIN_USER")
        admin_pass = os.getenv("ADMIN_PASSWORD")

        if admin_user and username == admin_user and password == admin_pass:
            session.clear()
            session["username"] = username
            session["role"] = "admin"
            session["csrf_token"] = secrets.token_urlsafe(32)
            session["gemini_ready"] = False
            return redirect(url_for("chatbot"))

        if login_user(username, password):
            session.clear()
            session["username"] = username
            session["role"] = "user"
            session["csrf_token"] = secrets.token_urlsafe(32)
            session["gemini_ready"] = False
            return redirect(url_for("chatbot"))

        return render_template("index.html", error=" Sai tên đăng nhập hoặc mật khẩu!")

    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"].strip()
        confirm = request.form["confirm"].strip()

        if not username or not email or not password or not confirm:
            return render_template("register.html", error="❌ Vui lòng nhập đầy đủ thông tin!")
        if not email.endswith(".edu") and not email.endswith(".edu.vn"):
            return render_template("register.html", error="❌ Chỉ chấp nhận email .edu hoặc .edu.vn!")
        if password != confirm:
            return render_template("register.html", error="❌ Hai mật khẩu không khớp!")

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username, email FROM Users WHERE username=? OR email=?", (username, email))
        existing_user = cursor.fetchone()
        conn.close()

        if existing_user:
            if existing_user[0] == username:
                return render_template("register.html", error=" Tên đăng nhập đã tồn tại!")
            if existing_user[1] == email:
                return render_template("register.html", error=" Email đã được sử dụng!")

        otp = str(random.randint(100000, 999999))
        try:
            msg = Message(
                subject="Mã xác thực tài khoản HUSC",
                sender=app.config["MAIL_USERNAME"],
                recipients=[email],
            )
            msg.body = (
                f"Xin chào {username},\n\n"
                f"Mã xác thực (OTP) của bạn là: {otp}\n"
                f"Mã này có hiệu lực trong 5 phút.\n\n"
                f"Trân trọng,\nKhoa CNTT - Đại học Khoa học Huế."
            )
            mail.send(msg)
        except Exception as e:
            print("❌ Lỗi khi gửi email:", e)
            return render_template("register.html", error="❌ Không thể gửi email xác thực. Vui lòng thử lại!")

        session["otp"] = otp
        session["pending_user"] = {"username": username, "email": email, "password": password}
        session["otp_time"] = time.time()
        flash("Đăng ký thành công! Vui lòng nhập mã OTP.")
        return redirect(url_for("verify_email"))

    return render_template("register.html")


@app.route("/verify_email", methods=["GET", "POST"])
def verify_email():
    if request.method == "POST":
        user_otp = request.form.get("otp")
        real_otp = session.get("otp")
        pending_user = session.get("pending_user")

        if not pending_user:
            flash("⚠️ Phiên đăng ký không hợp lệ, vui lòng đăng ký lại.", "error")
            return redirect(url_for("register"))

        otp_time = session.get("otp_time", 0)
        if time.time() - otp_time > 300:
            session.pop("otp", None)
            session.pop("pending_user", None)
            session.pop("otp_time", None)
            return render_template("verify_email.html", error="OTP expired. Please register again.")

        if user_otp == real_otp:
            add_userdata(pending_user["username"], pending_user["email"], pending_user["password"])
            session.pop("otp", None)
            session.pop("pending_user", None)
            session.pop("otp_time", None)
            flash("Đăng ký thành công. Vui lòng đăng nhập!", "success")
            return redirect(url_for("login"))

        return render_template("verify_email.html", error="❌ Mã xác thực không đúng!")

    return render_template("verify_email.html")


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM Users WHERE email=?", (email,))
        user_data = cursor.fetchone()
        conn.close()

        if not user_data:
            return render_template(
                "forgot_password.html",
                error="✅ Nếu email tồn tại, hệ thống đã gửi mã xác thực.",
            )

        username = user_data[0]
        otp = str(random.randint(100000, 999999))

        try:
            msg = Message(
                subject="Mã xác thực đặt lại mật khẩu - HUSC",
                sender=app.config["MAIL_USERNAME"],
                recipients=[email],
            )
            msg.body = (
                f"Xin chào {username},\n\n"
                f"Bạn đã yêu cầu đặt lại mật khẩu.\n"
                f"Mã xác thực (OTP) của bạn là: {otp}\n"
                f"Mã này có hiệu lực trong 5 phút.\n\n"
                f"Trân trọng,\nKhoa CNTT - Đại học Khoa học Huế."
            )
            mail.send(msg)
        except Exception as e:
            print("❌ Lỗi khi gửi email reset password:", e)
            return render_template("forgot_password.html", error="❌ Không thể gửi email. Vui lòng thử lại!")

        session["reset_otp"] = otp
        session["reset_email"] = email
        session["reset_username"] = username
        session["reset_otp_time"] = time.time()

        flash("✅ Gửi mã thành công. Vui lòng xác thực OTP!")
        return redirect(url_for("verify_reset_otp"))

    return render_template("forgot_password.html")


@app.route("/verify_reset_otp", methods=["GET", "POST"])
def verify_reset_otp():
    if request.method == "POST":
        user_otp = request.form.get("otp")
        real_otp = session.get("reset_otp")
        reset_email = session.get("reset_email")
        reset_username = session.get("reset_username")

        if not reset_email or not reset_username:
            flash("⚠️ Phiên reset mật khẩu không hợp lệ, vui lòng thực hiện lại.", "error")
            return redirect(url_for("forgot_password"))

        otp_time = session.get("reset_otp_time", 0)
        if time.time() - otp_time > 300:
            session.pop("reset_otp", None)
            session.pop("reset_email", None)
            session.pop("reset_username", None)
            session.pop("reset_otp_time", None)
            return render_template("verify_reset_otp.html", error="❌ Mã OTP đã hết hạn. Vui lòng yêu cầu lại!")

        if user_otp == real_otp:
            session["otp_verified"] = True
            flash("Xác thực thành công! Vui lòng đặt mật khẩu mới.", "success")
            return redirect(url_for("reset_password_form"))

        return render_template("verify_reset_otp.html", error="❌ Mã xác thực không đúng!")

    return render_template("verify_reset_otp.html")


@app.route("/reset_password_form", methods=["GET", "POST"])
def reset_password_form():
    if not session.get("otp_verified"):
        flash("⚠️ Vui lòng xác thực OTP trước!", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]
        username = session.get("reset_username")

        if not username:
            flash("⚠️ Phiên không hợp lệ!", "error")
            return redirect(url_for("forgot_password"))

        if new_password != confirm_password:
            return render_template("reset_password_form.html", error="❌ Hai mật khẩu không khớp!")

        if reset_password(username, new_password):
            session.pop("reset_otp", None)
            session.pop("reset_email", None)
            session.pop("reset_username", None)
            session.pop("reset_otp_time", None)
            session.pop("otp_verified", None)
            flash("✅ Đặt lại mật khẩu thành công! Vui lòng đăng nhập.", "success")
            return redirect(url_for("login"))

        return render_template("reset_password_form.html", error="❌ Lỗi khi đặt lại mật khẩu!")

    return render_template("reset_password_form.html")


@app.route("/chatbot")
def chatbot():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    history = get_chat_history(username)

    global gemini_ready
    status = fastapi_status()
    gemini_ready = bool(status.get("gemini_ready")) if "error" not in status else False
    session["gemini_ready"] = gemini_ready

    return render_template(
        "chatbot.html",
        username=username,
        messages=history,
        now=datetime.now(),
        gemini_ready=gemini_ready,
    )


@app.route("/chat", methods=["POST"])
def chat():
    if "username" not in session:
        return jsonify({"reply": "Vui lòng đăng nhập trước khi trò chuyện!"})

    username = session["username"]
    user_msg = (request.json or {}).get("message", "")
    if not user_msg.strip():
        return jsonify({"reply": "❗ Vui lòng nhập tin nhắn hợp lệ."})

    history_before = get_chat_history(username)
    recent_history = " ".join([h["content"] for h in history_before[-3:] if h["role"] == "user"])

    save_chat(username, "user", user_msg)

    try:
        data = fastapi_post("/chat", payload={"message": user_msg, "recent_history": recent_history}, timeout=180)
        bot_reply = (data or {}).get("reply") or "⚠️ Không nhận được phản hồi."
    except Exception as e:
        bot_reply = f"❌ FastAPI chat service unreachable: {e}"

    # === DEBUG: Log raw response để kiểm tra backend output ===
    print("\n" + "="*60)
    print("🔍 DEBUG RAW BOT REPLY:")
    print("="*60)
    print(bot_reply[:2000])  # In 2000 ký tự đầu
    print("="*60)
    print(f"📏 Total length: {len(bot_reply)} chars")
    print("="*60 + "\n")

    # Save bot reply in background thread to avoid blocking user
    threading.Thread(
        target=save_chat, 
        args=(username, "bot", bot_reply), 
        daemon=True
    ).start()

    response = Response(
        json.dumps({"reply": bot_reply}, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
    )
    return response


@app.route("/connect_api", methods=["POST"])
def connect_api():
    global gemini_ready
    try:
        fastapi_post("/connect_api", payload=None, timeout=30)
        gemini_ready = True
        session["gemini_ready"] = True
        return jsonify({"success": True, "message": "✅ Kết nối API thành công!"})
    except Exception as e:
        gemini_ready = False
        session["gemini_ready"] = False
        return jsonify({"success": False, "error": str(e)})


@app.route("/disconnect_api", methods=["POST"])
def disconnect_api():
    global gemini_ready
    try:
        fastapi_post("/disconnect_api", payload=None, timeout=10)
    except Exception:
        pass
    gemini_ready = False
    session["gemini_ready"] = False
    return jsonify({"success": True, "message": "✅ Đã ngắt kết nối API!"})


@app.route("/rebuild_db", methods=["POST"])
def rebuild_db():
    if session.get("role") != "admin":
        return jsonify({"success": False, "error": "Unauthorized: Admin access required"}), 403

    try:
        data = fastapi_post("/rebuild_db", payload=None, timeout=600)
        return jsonify(data)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    if "username" in session:
        username = session["username"]
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Chats WHERE username=?", (username,))
        conn.commit()
        conn.close()
    return jsonify({"success": True})


@app.route("/clear_history", methods=["POST"])
def clear_history():
    return clear_chat()


@app.route("/debug_status")
def debug_status():
    status = fastapi_status()
    status["session_username"] = session.get("username")
    status["session_gemini_ready"] = session.get("gemini_ready")
    return jsonify(status)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = get_local_ip()
    initialize_once()

    print("=" * 70)
    print("    🚀 FLASK CHATBOT SERVER - LAN ACCESS")
    print("=" * 70)
    print()
    print("🌐 Server đang khởi chạy tại:")
    print("   - Local:   http://127.0.0.1:5000")
    print(f"   - Network: http://{local_ip}:5000")
    print()
    print("🛑 Nhấn Ctrl+C để dừng server")
    print("=" * 70)
    print()

    debug_mode = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False,
    )
