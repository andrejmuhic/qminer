ECHO gluing visualization libraries
if not exist build\gui\js mkdir build\gui\js
copy /B src\qminer\gui\js\Highcharts\js\highcharts.js+src\qminer\gui\js\Highcharts\js\modules\exporting.js+src\qminer\js\visualization.js build\gui\js\visualization.js
