#!/bin/bash
file=${1%.svg}
echo "Converting $file.svg to $file.png"

#color=$( convert "$file.svg" -format "%[pixel:p{0,0}]" info:- )
#convert "$file.svg" -alpha off -bordercolor $color -border 1 \
#    \( +clone -fuzz 30% -fill none -floodfill +0+0 $color \
#       -alpha extract -geometry 200% -blur 0x0.5 \
#       -morphology erode square:1 -geometry 50% \) \
#    -compose CopyOpacity -composite -shave 1 "$file.png"

# convert -background none "$file.svg" "$file.png"
rsvg-convert -f png -o "$file.png" "$file.svg"
