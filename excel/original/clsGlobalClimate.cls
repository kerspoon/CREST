Attribute VB_Name = "clsGlobalClimate"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
' ===============================================================================================
' ## Global climate model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

' // Declare constants

' Day of the year that summer time starts
Private Const DAY_SUMMER_TIME_STARTS As Double = 87

' Day of the year that summer time ends
Private Const DAY_SUMMER_TIME_END As Double = 304
    

' // Declare variables
Public dblDayOfYear As Double
Public dblLongitude As Double
Public dblLatitude As Double
Public dblMeridian As Double
    
' An array to store the clearness index
Private aClearnessIndex(1 To 1440, 1 To 1) As Double

' An array to store the clear sky beam radiation at surface plane tracking sun (W/m^2)
Private aG_oClearsky(1 To 1440, 1 To 1) As Double

' An array to store the global surface horizontal radiation (W/m^2)
Private aG_o(1 To 1440, 1 To 1) As Double

Private aTempArray(1 To 365, 1 To 7) As Double

' An array to store the outside temperature (degrees C)
Private aTheta_o(1 To 1440, 1 To 1) As Double


' ===============================================================================================
' Class Properties

Public Property Get GetClearnessIndex() As Double()
    GetClearnessIndex = aClearnessIndex()
End Property

Public Property Get GetG_oClearsky() As Double()
    GetG_oClearsky = aG_oClearsky()
End Property

Public Property Get GetG_o() As Double()
    GetG_o = aG_o()
End Property

Public Property Get GetTheta_o() As Double()
    GetTheta_o = aTheta_o()
End Property

Public Property Get GetTempArray() As Double()
    GetTempArray = aTempArray()
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## SimulateClearnessIndex
'
' ===============================================================================================
Public Sub SimulateClearnessIndex()

    ' // Initialise the random number generator
    Randomize
    
    ' // Variables
    Dim intTimeStep As Integer
    Dim dblRand As Double
    Dim dblRand2 As Double
    Dim dblCumulativeP As Double
    Dim intRow As Integer
    Dim strCol As String
    Dim x As Integer
    Dim y As Integer
    
    ' // Read the input array
    Dim aKArray(1 To 101, 1 To 101) As Double
    For y = 1 To 101
        For x = 1 To 101
        
            ' // Get the column reference
            strCol = Cells(1, x + 2).Address(True, False, xlA1)
            strCol = Left(strCol, InStr(strCol, "$") - 1)
            
            ' // Store the data
            aKArray(y, x) = wsClearnessIndexTPM.Range(strCol + CStr(y + 9)).Value
        Next x
    Next y
    
    ' // Set the starting value
    ' // Bin range is 1 to 101, where 101 is clear sky
    Dim iCurrentStateBin As Integer
    
    ' // Start with a clear sky
    iCurrentStateBin = 101
    
    ' // Store the starting k value
    aClearnessIndex(1, 1) = 1
    
    ' // Now work out the transition steps
    For intTimeStep = 2 To 1440
           
        ' // Get a random number
        dblRand = Rnd()
    
        ' // Reset the cumulative probability count
        dblCumulativeP = 0
    
        ' // Define a loop count variable
        Dim i As Integer
    
        ' // Cycle through the probabilities for this state
        For i = 1 To 101
                             
            ' // Add this probability
            dblCumulativeP = dblCumulativeP + aKArray(iCurrentStateBin, i)
                       
            ' // See if this is a state transition
            If dblRand <= dblCumulativeP Then
            
                ' // Transition to another or same state
                iCurrentStateBin = i
                Exit For
            
            End If
        Next i
       
        ' // Work out the k value for this step
        Dim dk As Double
        If iCurrentStateBin = 101 Then
            dk = 1
        Else
            dk = (iCurrentStateBin / 100) - 0.01
        End If
        
        ' // Store the k value
        aClearnessIndex(intTimeStep, 1) = dk
        
    Next intTimeStep
   
End Sub




' ===============================================================================================
' ## CalculateGlobalIrradiance
'
' ===============================================================================================
Public Sub CalculateGlobalIrradiance()
    
    
    ' // Declare variables
    ' a count variable to iterate through the minutes of the day
    Dim intCount As Integer
    
    ' a loop variable to iterate through the hours of the day
    Dim intHour As Integer
    
    ' a loop variable to iterate through the minutes of the hour
    Dim intMinute As Integer
    
    ' direct beam radiation on panel
    Dim dblG_direct As Double
    
    ' diffuse radiation on panel
    Dim dblG_diffuse As Double
    
    ' reflected radiation on panel
    Dim dblG_reflected As Double
    
    ' clearness index
    Dim dblClearnessIndex As Double
    
    ' solar incident angle on panel
    Dim dblSolarIncidentAngle As Double
    
    ' clear sky beam radiation at surface (horizontal)
    Dim dblG_oClearsky As Double
    
    ' Extraterrestrial radiation
    Dim dblG_et As Double
    
    ' optical depth
    Dim dblOpticalDepth As Double
    
    ' solar altitude angle
    Dim dblSolarAltitudeAngle As Double
    
    Dim dblAzimuthOfSun As Double
    
    Dim dblAzimuthAngleTest As Double
    
    ' adjusted azimute of sun
    Dim dblAdjustedAzimuthOfSun As Double
    
    Dim dblHourAngle As Double
    
    Dim dblDeclination As Double
    
    Dim dblHoursBeforeSolarNoon As Double
    
    Dim dblTimeCorrectionFactor As Double
    
    Dim dblEquationOfTime As Double
    
    Dim dblB As Double
    
    Dim intLocalStandardTimeHour As Integer
    
    Dim intLocalStandardTimeMinute As Integer
    
    Dim intDayOfMonth As Integer
    
    Dim intMonthOfYear As Integer
    
    Dim strDate As String
        
    Dim dblSkyDiffuseFactor As Double
    
    ' // Get longitude, latitude and meridian
    dblLongitude = wsMain.Range("rLongitude").Value
    dblLatitude = wsMain.Range("rLatitude").Value
    dblMeridian = wsMain.Range("rMeridian").Value
           
    ' // Get day of year
    intDayOfMonth = wsMain.Range("rDayOfMonth").Value
    intMonthOfYear = wsMain.Range("rMonthOfYear").Value
    strDate = CStr(intDayOfMonth) + "/" + CStr(intMonthOfYear) + "/2015" ' <--- year value is arbitrary and not relevant to the calculations
    dblDayOfYear = DatePart("y", CDate(strDate))
        
    ' // Calculate B
    dblB = 360 * (dblDayOfYear - 81) / 364
    
    ' // Calculate equation of time
    dblEquationOfTime = (9.87 * Sin(2 * dblB * PI / 180)) - (7.53 * Cos(dblB * PI / 180)) - (1.5 * Sin(dblB * PI / 180))
    
    ' // Calculate time correction factor
    dblTimeCorrectionFactor = (4 * (dblLongitude - dblMeridian)) + dblEquationOfTime
    
    ' // Start the count variable
    intCount = 1
    
    ' // Loop through the hours of the day
    For intHour = 0 To 23
    
        ' // Update local standard time hour but only if Daylight saving applies in that country
        ' If wsMain.Shapes("objDaylightSaving").ControlFormat.Value = 1 Then
        If ((dblDayOfYear >= DAY_SUMMER_TIME_STARTS And dblDayOfYear < DAY_SUMMER_TIME_END) _
            And wsMain.Shapes("objDaylightSaving").ControlFormat.Value = 1) Then
            intLocalStandardTimeHour = intHour - 1
        Else
            intLocalStandardTimeHour = intHour
        End If
        ' End If
                
        ' // Loop through the minutes of the day
        For intMinute = 1 To 60
        
            ' // update local standard time minute
            intLocalStandardTimeMinute = intMinute
            
            ' // Calculate hours before solar noon
            dblHoursBeforeSolarNoon = 12 - (intLocalStandardTimeHour + (intLocalStandardTimeMinute / 60) + (dblTimeCorrectionFactor / 60))
            
            ' // Calculate extraterrestrial radiation
            dblG_et = 1367 * (1 + (0.034 * Cos(2 * PI * dblDayOfYear / 365.25)))
                    
            ' // Calculate optical depth
            dblOpticalDepth = 0.174 + (0.035 * Sin(2 * PI * (dblDayOfYear - 100) / 365))
                        
            ' // Calculate hour angle
            dblHourAngle = 15 * dblHoursBeforeSolarNoon
            
            ' // Calculate declination
            dblDeclination = 23.45 * Sin(2 * PI * (284 + dblDayOfYear) / 365.25)
            
            ' // Calculate solar altitude angle
            dblSolarAltitudeAngle = 180 / PI * Application.WorksheetFunction.Asin((Cos(dblLatitude * PI / 180) * Cos(dblDeclination * PI / 180) * Cos(dblHourAngle * PI / 180)) + (Sin(dblLatitude * PI / 180) * Sin(dblDeclination * PI / 180)))
            
            ' // Calculate clearsky beam radiation at surface (plane tracking the sun)
            ' // NOTE: in previous versions this was incorrectly called clearsky beam radiation at surface (horizontal)
            If dblSolarAltitudeAngle > 0 Then
                aG_oClearsky(intCount, 1) = dblG_et * Exp((0 - dblOpticalDepth) / Sin(dblSolarAltitudeAngle * PI / 180))
            Else
                aG_oClearsky(intCount, 1) = 0
            End If
            
            ' // Get clearness index for this time step
            dblClearnessIndex = aClearnessIndex(intCount, 1)
            
            ' // Calculate global radiation on surface horizontal
            ' // NOTE: in previous versions this was calculated incorrectly and actually calculated global radiation (plane tracking the sun)
            aG_o(intCount, 1) = aG_oClearsky(intCount, 1) * dblClearnessIndex * Sin(dblSolarAltitudeAngle * PI / 180)
                        
            ' // Iterate the count variable
            intCount = intCount + 1
            
        Next intMinute
        
    Next intHour

End Sub




' ===============================================================================================
' ## RunTemperatureModel
'
' ===============================================================================================
Public Sub RunTemperatureModel()

    Randomize
    
    ' // Generate stochastic average daily temperatures for a whole year
    Call Td_model
    
    ' // Declare variables
        
    ' A loop variable (for the minute of the day)
    Dim i As Integer
    
    ' A loop variable (for array columns)
    Dim j As Integer
    
    ' extraterrestrial irradiation
    Dim cSolarConstant As Integer
    
    ' A day of the year variable
    Dim DoY As Integer
    
    ' The minute of sunset
    Dim Td_sunset_i As Integer
    
    ' The minute of minimum temperature
    Dim Td_min_i As Integer
    
    ' A time interval in minutes
    Dim di As Integer
    
    ' The maximum of the ratio of global horizontal irradiance to extraterrestrical irradiance
    Dim kx_max As Double
    
    ' The minute of the day with the above maximum occurs
    Dim kx_max_i As Double
    
    ' The average daily temperature for the day of year chosen
    Dim Td As Double
    
    ' The minimum temperature of the day
    Dim Td_min As Double
    
    ' The maximum temperature of the day
    Dim Td_max As Double
    
    ' The temperature at sunset
    Dim Td_sunset As Double
    
    ' A temperature difference
    Dim dTd As Double
    
    ' A temperature standard deviation
    Dim dTd_SD As Double
    
    ' A variable storing the cumulative global horizontal irradiance
    Dim Irradiation As Double
    
    ' Slope variables
    Dim slope_before As Double
    Dim slope_after As Double
    
    ' A temperature difference variable
    Dim dT As Double
    
    ' A temperature cooling rate variable
    Dim CoolingRate As Double
    
    ' The mean clearness index overnight
    Dim vOvernightMeanClearness As Double
    
    ' The cloud cooling rate
    Dim CloudCoolingRate As Double
    
    ' an array to store the simulation output
    Dim vCumulativeRadiationArray(1 To 1440, 1 To 4) As Variant
    
    ' Flag to indicate that sun never rises in the Arctic Night
    Dim blnArcticNight As Boolean
    blnArcticNight = False
      
    ' // Set the day of year
    DoY = dblDayOfYear
    
    ' // determine the extraterrestrial irradiance
    cSolarConstant = 1367 * (1 + (0.034 * Cos(2 * WorksheetFunction.PI * DoY / 365.25)))
    
    ' // set the ratio of global horizontal irradiance to extraterrestrical irradiance to zero
    kx_max = 0
    kx_max_i = 0
    
    ' // Get the daily average temperature for this day of the year
    Td = aTempArray(DoY, 7)
    
    ' // Set the cloud cooling rate
    CloudCoolingRate = 0.1 / 60
        
    ' // Set the first values of the cumulative radiation array to zero
    vCumulativeRadiationArray(1, 1) = 0
    vCumulativeRadiationArray(1, 2) = 0
    vCumulativeRadiationArray(1, 3) = Null
        
    ' // Loop through each minute of the day
    For i = 2 To 1440
        ' // calculate the cumulative global horizontal irradiance
        vCumulativeRadiationArray(i, 2) = vCumulativeRadiationArray(i - 1, 2) + aG_o(i, 1)
        
        ' // if it is daylight, calculate the cumulative extraterrestrial irradiance and calculate a ratio of irradiances
        If aG_o(i, 1) > 0 Then
            vCumulativeRadiationArray(i, 1) = vCumulativeRadiationArray(i - 1, 1) + cSolarConstant
            vCumulativeRadiationArray(i, 3) = vCumulativeRadiationArray(i, 2) / vCumulativeRadiationArray(i, 1)
        Else
            vCumulativeRadiationArray(i, 1) = 0
            vCumulativeRadiationArray(i, 3) = Null
        End If
                        
        ' // Find the maximum of the ratio and store the minute when this occurs
        If vCumulativeRadiationArray(i, 3) > kx_max Then
            kx_max = vCumulativeRadiationArray(i, 3)
            kx_max_i = i
        End If
        
        ' // store the cumulative irradiance value
        If vCumulativeRadiationArray(i, 2) > Irradiation Then
            Irradiation = vCumulativeRadiationArray(i, 2)
        End If
    
    Next i
    
    ' // Convert units of irradiance
    Irradiation = (Irradiation / 1000) * (60 / 3600)
    
    ' // Set a temperature standard deviation
    dTd_SD = 1
    
    ' // Given the cumulative irradiance, determine the total deviation of temperatures around the daily average
    dTd = 20 * Application.WorksheetFunction.Log(Irradiation + 2.5, 10) - 7
    
    dTd = dTd + Application.WorksheetFunction.NormInv(Rnd(), 0, dTd_SD)
    
    ' // Set the max and min temperature for the day
    Td_min = Td - 0.5 * dTd
    Td_max = Td + 0.5 * dTd
    
    ' //Workaround for cases in winter inside the Arctic Circle where radiation is zero all day
    If kx_max = 0 Then
        kx_max = 1
        kx_max_i = 720
        Td_min_i = 1
        blnArcticNight = True
    End If
         
    ' // Set the slopes of the temperature profiles before and after the maximum temperature
    slope_before = (Td_max - Td_min) / kx_max
    slope_after = slope_before * 1.7
    
    ' // For each minute up to the maximum temperature
    For i = 2 To kx_max_i
        ' // If it's daylight, set the external air temperature
        If aG_o(i, 1) > 0 Then
            vCumulativeRadiationArray(i, 4) = Td_min + slope_before * vCumulativeRadiationArray(i, 3)
            aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
        ElseIf blnArcticNight Then
            vCumulativeRadiationArray(i, 4) = Td_min + slope_before * i / kx_max_i
            aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
        Else
            vCumulativeRadiationArray(i, 4) = Null
            Td_min_i = i + 1
        End If
    Next i
    
    ' // Default sunset time is midday (for Arctic night)
    If blnArcticNight Then
        Td_sunset_i = 1080
        For i = kx_max_i To Td_sunset_i
            vCumulativeRadiationArray(i, 4) = Td_max - slope_before * (i - kx_max_i) / kx_max_i
            aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
            Td_sunset = vCumulativeRadiationArray(i, 4)
        Next i
    Else
    ' // For each minute after the maximum temperature
        For i = kx_max_i + 1 To 1439
            ' // If it's daylight, set the external air temperature
            If aG_o(i, 1) > 0 Then
                vCumulativeRadiationArray(i, 4) = Td_max - (slope_after * (kx_max - vCumulativeRadiationArray(i, 3)))
                aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
                Td_sunset = vCumulativeRadiationArray(i, 4)
                Td_sunset_i = i
            Else
                vCumulativeRadiationArray(i, 4) = Null
            End If
        Next i
    End If
    
    ' // Determine number of minutes outside of daylight
    di = Td_min_i + 1440 - Td_sunset_i
    
    ' // Determine the temperature difference between sunset and the minimum temperature for the day
    dT = Td_sunset - Td_min
    
    ' // Determine how much cooling is therefore required
    CoolingRate = dT / di
    CloudCoolingRate = 0.025
    
    '// Calculate the overnight mean clearness index
    For i = 1 To Td_min_i
        vOvernightMeanClearness = vOvernightMeanClearness + aClearnessIndex(i, 1)
    Next i
    For i = Td_sunset_i To 1440
        vOvernightMeanClearness = vOvernightMeanClearness + aClearnessIndex(i, 1)
    Next i
    vOvernightMeanClearness = vOvernightMeanClearness / di
        
    ' // Calculate the overnight temperatures
    For i = Td_sunset_i To 1440
        vCumulativeRadiationArray(i, 4) = vCumulativeRadiationArray(i - 1, 4) - (CoolingRate - CloudCoolingRate * (vOvernightMeanClearness - aClearnessIndex(i, 1)))
        aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
        
    Next i
    
    vCumulativeRadiationArray(1, 4) = vCumulativeRadiationArray(1440, 4)
    aTheta_o(1, 1) = vCumulativeRadiationArray(1, 4)
        
    For i = 2 To Td_min_i
        vCumulativeRadiationArray(i, 4) = vCumulativeRadiationArray(i - 1, 4) - (CoolingRate - CloudCoolingRate * (vOvernightMeanClearness - aClearnessIndex(i, 1)))
        aTheta_o(i, 1) = vCumulativeRadiationArray(i, 4)
    Next i
    
    '// Output temperature data for inspection
    For i = 1 To 1440
        For j = 1 To 4
            ThisWorkbook.Worksheets("Irradiance").Cells(i + 3, 13 + j) = vCumulativeRadiationArray(i, j)
        Next j
    Next i
End Sub




' ===============================================================================================
' ## Td_model
'
' ===============================================================================================

Private Sub Td_model()
' Private Sub Td_model()
      
    ' Cities for temperature climate, read from main worksheet
    Dim blnEngland As Boolean
    Dim blnN_Delhi As Boolean
    Dim blnMumbai As Boolean
    Dim blnBengaluru As Boolean
    Dim blnChennai As Boolean
    Dim blnKolkata As Boolean
    Dim blnItanagar As Boolean
    ' Monthly average temperatures
    Dim aMonthlyTemps() As Variant ' declare an unallocated array.
    ' Mean temperature for the year
    Dim Ty_mean As Double
    ' Standard deviation of temperatures for the year
    Dim Ty_SD As Double
    ' Day offset to shift sinusoidal approximation to daily temperature profile
    Dim Ty_shift As Double
    ' Pi
    Dim PI As Double
    ' Convert from radians to degrees
    ' Dim rad2deg As Double
    ' Autoregression variable
    Dim AR As Double
    ' Moving average variable
    Dim MA As Double
    ' Standard deviation variable to scale max-min variation in each season
    Dim SDFactor As Double
    ' Resuling standard deviation of temperature in each month
    Dim SD
    ' Month of year from 1 to 12
    Dim Month As Integer
    ' A loop variable
    Dim i As Integer
    
    ' Select Location for temperature climate
    blnEngland = IIf(wsMain.Range("rCity").Value = "England", True, False)
    blnN_Delhi = IIf(wsMain.Range("rCity").Value = "N Delhi", True, False)
    blnMumbai = IIf(wsMain.Range("rCity").Value = "Mumbai", True, False)
    blnBengaluru = IIf(wsMain.Range("rCity").Value = "Bengaluru", True, False)
    blnChennai = IIf(wsMain.Range("rCity").Value = "Chennai", True, False)
    blnKolkata = IIf(wsMain.Range("rCity").Value = "Kolkata", True, False)
    blnItanagar = IIf(wsMain.Range("rCity").Value = "Itanagar", True, False)
    ' Read in monthly average temperatures from Climate Data worksheet
    If blnEngland Then
        aMonthlyTemps = wsClimateData.Range("rEnglandTemps")
    ElseIf blnN_Delhi Then
        aMonthlyTemps = wsClimateData.Range("rNDelhiTemps")
        ElseIf blnMumbai Then
        aMonthlyTemps = wsClimateData.Range("rMumbaiTemps")
        ElseIf blnBengaluru Then
        aMonthlyTemps = wsClimateData.Range("rBengaluruTemps")
        ElseIf blnChennai Then
        aMonthlyTemps = wsClimateData.Range("rChennaiTemps")
        ElseIf blnKolkata Then
        aMonthlyTemps = wsClimateData.Range("rKolkataTemps")
        ElseIf blnItanagar Then
        aMonthlyTemps = wsClimateData.Range("rItanagarTemps")
        Else
        wsMain.Range("J18") = "Error - simulation stopped"
        MsgBox "Please choose a region or city with temperature data from the ClimateData&CoolingTech worksheet", vbExclamation
        End
    End If
    
    ' // Set the statistics that characterise long term daily average temperature for central England (based on HadCet data)
    Ty_mean = 9.3
    Ty_SD = 6.5
    Ty_shift = -115
    AR = 0.81
    MA = 0.62
    SDFactor = 0.1
    
    
    PI = 3.14159265359
    ' rad2deg = PI / 180
    
    ' // Determine sinusoidal approximation to daily average temperatures
    For i = 1 To 365
        If i < 32 Then
        Month = 1
        ElseIf i < 60 Then
        Month = 2
        ElseIf i < 91 Then
        Month = 3
        ElseIf i < 121 Then
        Month = 4
        ElseIf i < 152 Then
        Month = 5
        ElseIf i < 182 Then
        Month = 6
        ElseIf i < 213 Then
        Month = 7
        ElseIf i < 244 Then
        Month = 8
        ElseIf i < 274 Then
        Month = 9
        ElseIf i < 305 Then
        Month = 10
        ElseIf i < 335 Then
        Month = 11
        Else
        Month = 12
        End If
        aTempArray(i, 1) = aMonthlyTemps(Month, 1)
        SD = (aMonthlyTemps(Month, 3) - aMonthlyTemps(Month, 2)) * SDFactor
        ' aTempArray(i, 1) = Ty_mean + Ty_SD * Sin(360 * (i / 365) * rad2deg + Ty_shift / 365 * 2 * PI)
        aTempArray(i, 2) = Application.WorksheetFunction.NormInv(Rnd(), 0, SD)
    Next i
    
    ' // Set values of the 1st parameters for AR, MA ,ARMA
    aTempArray(1, 3) = 0
    aTempArray(1, 4) = 0
    aTempArray(1, 5) = 0
    
    ' // Use ARMA model to add appropriate randomness to daily average temperatures
    ' // Calculate...
    For i = 2 To 365
        ' //...AR part AR(t)= AR*AR(t-1)+E(t)
        aTempArray(i, 3) = aTempArray(i - 1, 3) * AR + aTempArray(i, 2)
        ' //... MA part: MA= MA*E(t-1)+E(t)
        aTempArray(i, 4) = aTempArray(i, 2) + aTempArray(i - 1, 2) * MA
        ' //... ARMA part: ARMA= AR*AR(t-1)+MA*E(t-1)+E(t)
        aTempArray(i, 5) = aTempArray(i - 1, 3) * AR + aTempArray(i - 1, 2) * MA + aTempArray(i, 2)
    Next i
    
    ' // Calculate Td_model_Ar and Td_model_ARMA
    For i = 1 To 365
        aTempArray(i, 6) = aTempArray(i, 1) + aTempArray(i, 3)
        aTempArray(i, 7) = aTempArray(i, 1) + aTempArray(i, 5)
    Next i
        
End Sub



' ===============================================================================================
' ## WriteGlobalClimate
' Write the global environmental variables to the worksheet
' ===============================================================================================
Public Sub WriteGlobalClimate()
    
    ' // Write the simulation data to the sheet
    With wsGlobalClimate
        .Range("B5:B1444") = aClearnessIndex
        .Range("C5:C1444") = aG_oClearsky
        .Range("D5:D1444") = aG_o
        .Range("E5:E1444") = aTheta_o
        .Range("J4.P368") = aTempArray()
    End With
End Sub


