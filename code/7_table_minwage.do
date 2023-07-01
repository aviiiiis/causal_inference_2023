clear all
set more off
cap log close
global sysdate = c(current_date)
graph set window fontface "Times New Roman"

log using "${log}\7_table_minwage.smcl", replace


use "${rdata}\minw.dta", clear

graph twoway (scatter minimumwagemonth year, msymbol(circle_hollow) mcolor(black)) ///
	(scatter minimumwagemonth year if year == 2007), ///
	ytitle("Minimum wage (monthly)") xtitle("Year") ///
	xlabel(1965(5)2025, labsize(*0.75)) ylabel(,labsize(*0.75)) ///
	legend(off) graphregion(color(white))
graph export "${pic}\minw_mon.png", replace


drop if year < 1992
graph twoway (scatter minimumwagehour year, msymbol(circle_hollow) mcolor(black)) ///
	(scatter minimumwagehour year if year == 2007), ///
	ytitle("Minimum wage (hourly)") xtitle("Year") ///
	xlabel(1990(5)2025, labsize(*0.75)) ylabel(,labsize(*0.75)) ///
	legend(off) graphregion(color(white))
graph export "${pic}\minw_hr.png", replace


estpost tabstat minimumwagemonth minimumwagehour, by(year)
esttab using "${table}\minw.tex", cells("minimumwagemonth minimumwagehour") noobs ///
nonumber varlabels(`e(labels)')  ///
drop(Total) varwidth(30) title(Minimum wage from 1992 to 2023 in Taiwan\label{tab:minw}) ///
 collab("Min. wage (month)" "Min. wage (hour)", lhs("year")) tex replace

log close