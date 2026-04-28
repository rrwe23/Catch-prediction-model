import folium

m = folium.Map(location=[33.4, 126.5], zoom_start=7)

# 조업 범위 사각형
folium.Rectangle(
    bounds=[[32.0, 124.5], [34.5, 128.5]],
    color="blue",
    fill=True,
    fill_opacity=0.1
).add_to(m)

m.save("jeju_zone.html")  # 브라우저에서 열기