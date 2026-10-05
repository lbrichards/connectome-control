"""Render favicon fallbacks ONCE from the favicon.svg geometry (light
colours), committed as static files. Geometry duplicated exactly from the
SVG: cart rect(14,42,36x14,rx2,stroke5), pole line(32,42)-(32,17) w5 round
caps, tip circle(32,12,r7). Colours: ink #222222, accent #2B6CB0."""
from PIL import Image, ImageDraw

INK, ACCENT = (0x22, 0x22, 0x22, 255), (0x2B, 0x6C, 0xB0, 255)

def render(canvas_px):
    S = canvas_px / 64.0
    ss = 8                                   # supersample
    P = int(canvas_px * ss)
    s = S * ss
    img = Image.new("RGBA", (P, P), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # cart: rounded-rect OUTLINE, stroke 5 centred on the path
    d.rounded_rectangle([14*s, 42*s, (14+36)*s, (42+14)*s], radius=2*s,
                        outline=INK, width=round(5*s))
    # pole: width-5 line with round caps
    d.line([32*s, 42*s, 32*s, 17*s], fill=INK, width=round(5*s))
    for y in (42, 17):
        d.ellipse([32*s-2.5*s, y*s-2.5*s, 32*s+2.5*s, y*s+2.5*s], fill=INK)
    # tip
    d.ellipse([(32-7)*s, (12-7)*s, (32+7)*s, (12+7)*s], fill=ACCENT)
    return img.resize((canvas_px, canvas_px), Image.LANCZOS)

# favicon.ico: 16 + 32, transparent
i32, i16 = render(32), render(16)
i32.save("web/public/favicon.ico", sizes=[(16, 16), (32, 32)],
         append_images=[i16])

# apple-touch-icon: 180x180, OPAQUE white, glyph ~70% centred
glyph = render(126)                          # 70% of 180
apple = Image.new("RGB", (180, 180), (255, 255, 255))
apple.paste(glyph, ((180-126)//2, (180-126)//2), glyph)
apple.save("web/public/apple-touch-icon.png")
print("favicon.ico (16+32, transparent) and apple-touch-icon.png (180, white) written")
