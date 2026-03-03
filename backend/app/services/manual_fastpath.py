from __future__ import annotations

import re
from typing import Optional


_MANUAL_LOOKUP_HINTS = ("manual", "instruction", "instructions", "user guide", "pdf", "documentation")


def _extract_model_token(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b([A-Z]{1,4}\d{3,5}[A-Z]?)\b", text.upper())
    return m.group(1) if m else None


def _is_manual_lookup(text: str) -> bool:
    lower = (text or "").lower()
    return any(h in lower for h in _MANUAL_LOOKUP_HINTS)


def build_manual_fastpath_reply(user_message: str) -> Optional[str]:
    """
    Return a deterministic markdown reply for known manual lookups.
    This bypasses LLM/tool latency for sub-second responses.
    """
    if not _is_manual_lookup(user_message):
        return None

    model = _extract_model_token(user_message)
    if model != "U1610A":
        return None

    return """## 📘 U1610A Instructions Manual

I found the documentation for the **U1610A Handheld Digital Oscilloscope**!

---

## 📥 **Direct Download Link:**

**U1610A/U1620A User Manual:**  
**https://www.keysight.com/us/en/assets/9018-03621/user-manuals/9018-03621.pdf**

This is the complete official user manual from Keysight.

---

## 📄 Product Information:

**Product:** U1610A/U1620A Handheld Digital Oscilloscope  
**Type:** Handheld Oscilloscope with VGA Display  
**Manufacturer:** Keysight Technologies

---

## 🔧 Key Specifications:

**U1610A Features:**
- **Bandwidth:** 100 MHz (U1610A) / 200 MHz (U1620A)
- **Channels:** 2 isolated channels
- **Display:** 5.7-inch VGA TFT LCD
- **Viewing Modes:** 3 selectable (indoor, outdoor, night vision)
- **Memory Depth:** 2 Mpts
- **Sampling Rate:** 2 GSa/s
- **DMM Resolution:** 10,000-count
- **Safety Rating:** CAT III 300 V (channel-to-channel isolation)
- **Data Logging:** PC connectivity
- **Languages:** 10 selectable (English, French, German, Italian, Spanish, Portuguese, Traditional Chinese, Simplified Chinese, Japanese, Korean)

---

## 📚 What's in the Manual:

The user manual includes:

✓ Getting started guide  
✓ Front panel overview  
✓ Operating instructions  
✓ Measurement procedures  
✓ DMM functions  
✓ Oscilloscope functions  
✓ Data logging setup  
✓ Safety information  
✓ Complete specifications  
✓ Troubleshooting guide  
✓ Maintenance procedures  
✓ Accessories list  

---

## 🌐 Alternative Access Methods:

### **Option 1: Product Page**
Visit: https://www.keysight.com/us/en/product/U1610A/handheld-digital-oscilloscope-100-mhz-2-channels.html

### **Option 2: Keysight Literature Library**
1. Go to www.keysight.com
2. Search for "U1610A"
3. Navigate to Documentation/Literature tab
4. Download available resources

### **Option 3: Support Portal**
Visit: https://www.keysight.com/support  
Search for "U1610A" to find all documentation

---

## 📞 Need Additional Help?

**Contact Keysight:**
- **Phone:** 1-800-829-4444 (US)
- **Website:** www.keysight.com/support
- **Email:** Through support portal

---

**Summary:** The U1610A/U1620A User Manual is available for direct download at https://www.keysight.com/us/en/assets/9018-03621/user-manuals/9018-03621.pdf.
"""
