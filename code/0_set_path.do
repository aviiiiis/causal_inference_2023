
clear all
set more off
cap log close
global sysdate = c(current_date)


if "`c(username)'" == "ttyang" {
	
	global do = "D:\nest\Dropbox\RA_research\China_Xi\data\do"
    global rdata = "D:\nest\Dropbox\RA_research\China_Xi\data\rdata"
	global wdata = "D:\nest\Dropbox\RA_research\China_Xi\data\wdata"
    
}
if "`c(username)'" == "avis1" {
    
	global do = "D:\My_Drive\MA_semester04\Tzu\paper\do"
    global rdata = "D:\My_Drive\MA_semester04\Tzu\paper\rdata"
	global wdata = "D:\My_Drive\MA_semester04\Tzu\paper\wdata"
	global pic = "D:\My_Drive\MA_semester04\Tzu\paper\pic"
	global table = "D:\My_Drive\MA_semester04\Tzu\paper\table"
	global log = "D:\My_Drive\MA_semester04\Tzu\paper\log"
	
}
if "`c(username)'" == "" {
    
    global do = ""
    global rdata = ""
	global wdata = ""
	
}

if "`c(username)'" == "" {

    global do = ""
    global rdata = ""
	global wdata = ""

	
}

