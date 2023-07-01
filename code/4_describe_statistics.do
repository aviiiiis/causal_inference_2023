log using "${log}\4_describe_statistics.do.smcl", replace
clear all

use "${wdata}\mu_98-10", clear
estpost summarize age gender eduyr kid marital taipei hour earn_ad, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn_ad "earning") ///
		title(MUS: 1998-2010\label{tab:mus9810}) nonumber noobs  replace

use "${wdata}\mu_ML", clear
estpost summarize age gender eduyr kid marital taipei hour earn_ad, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status" earn_ad "earning") ///
		title(ML Dataset\label{tab:train}) nonumber noobs  append


use "${wdata}\mu_PD", clear
drop if year < 2005
estpost summarize age gender eduyr kid marital taipei, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max") ///
		coeflabels(eduyr "education year" marital "marital status") ///
		title(Prediction Dataset\label{tab:prediction}) nonumber noobs  append

/**

use "${wdata}\mu_00-10", clear
estpost summarize age gender eduyr marital city hour earn earn_hr, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(MUS: 2000-2010) nonumber replace

use "${wdata}\mu_00-07", clear
estpost summarize age gender eduyr marital city hour earn earn_hr, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(MUS: 2000-2007) nonumber append

use "${wdata}\mu_ML", clear
estpost summarize age gender eduyr marital city hour earn earn_hr, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(Training Data Set and Validation Data Set) nonumber append


use "${wdata}\mu_ML", clear
estpost summarize age gender eduyr marital city hour earn earn_hr if treat == 0, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(Training Data Set and Validation Data Set: Treatment group) nonumber append

use "${wdata}\mu_ML", clear
estpost summarize age gender eduyr marital city hour earn earn_hr if treat == 1, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(Training Data Set and Validation Data Set: Control group) nonumber append


use "${wdata}\mu_PD", clear
estpost summarize age gender eduyr marital city treat, detail
esttab using "${table}\desc.tex", cells("count mean(fmt(2)) sd(fmt(2)) min max p25 p50 p75") title(Prediction) nonumber append

**/
log close