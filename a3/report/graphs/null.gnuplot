set xlabel 'Null Likelihood'
set ylabel 'AER'
set term postscript enhanced 18
set output 'null.eps'
set key left top 
set size 0.8,0.8
#set size 0.7, 0.7
set yrange [0.35:0.37]
set xrange [0.05:0.40]

plot 'null.dat' using 2:7 notitle w linespoints


