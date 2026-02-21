
---

## 🚀 مراحل آپلود روی Hugging Face Spaces

### مرحله ۱: ساخت اکانت
1. به [huggingface.co](https://huggingface.co) بروید
2. با ایمیل ثبت‌نام کنید (Gmail یا Outlook)
3. ایمیل را تأیید کنید

### مرحله ۲: ساخت Space جدید
1. روی **New** → **Space** کلیک کنید
2. **Space name**: `silver-price-tracker` (یا هر نام دلخواه)
3. **SDK**: **Streamlit** را انتخاب کنید
4. **Space hardware**: CPU (رایگان)
5. **Public** بگذارید (Private نیاز به پرداخت دارد)
6. **Create Space**

### مرحله ۳: آپلود فایل‌ها
**روش ۱: از طریق وب**
1. وارد Space شوید
2. روی **Files** → **Upload files** کلیک کنید
3. سه فایل (`app.py`, `requirements.txt`, `README.md`) را آپلود کنید

**روش ۲: از طریق Git**
```bash
# نصب Git LFS (اگر حجم فایل‌ها زیاد است)
git lfs install

# کلون ریپازیتوری
git clone https://huggingface.co/spaces/YOUR_USERNAME/silver-price-tracker
cd silver-price-tracker

# کپی فایل‌ها
cp /path/to/your/app.py .
cp /path/to/your/requirements.txt .
cp /path/to/your/README.md .

# کامیت و پوش
git add .
git commit -m "Initial commit"
git push
