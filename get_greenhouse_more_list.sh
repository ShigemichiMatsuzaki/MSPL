for d in `ls /media/data/dataset/matsuzaki/greenhouse/color/ | grep 26`; do
  for f in `ls /media/data/dataset/matsuzaki/greenhouse/color/$d | grep png`; do
    if [ -e /media/data/dataset/matsuzaki/greenhouse/depth/$d/$f ]; then
      echo "/tmp/dataset/greenhouse/color/$d/$f,/tmp/dataset/greenhouse/trainannot/trav_gt/26_0_000000.png,/tmp/dataset/greenhouse/depth/$d/$f"  >> ./vision_datasets/greenhouse/train_greenhouse_more.txt
    else
      echo "$d/$f not found"
    fi
  done
done
