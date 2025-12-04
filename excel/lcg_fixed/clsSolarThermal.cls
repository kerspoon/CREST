Attribute VB_Name = "clsSolarThermal"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
' ===============================================================================================
' ## Solar thermal collector model
' ===============================================================================================




' Class Variables

' // Declare constants

' Day of the year that summer time starts
Private Const DAY_SUMMER_TIME_STARTS As Double = 87

' Day of the year that summer time ends
Private Const DAY_SUMMER_TIME_END As Double = 304

' Specify the ground reflectance constant
Private Const GROUND_REFLECTANCE As Double = 0.2

' // Index variables
Private intDwellingIndex As Integer

Private intSolarThermalIndex As Integer
Private intRunNumber As Integer
 
' // Collector variables

' number of collectors
Private dblN As Double

' size of collector absorber
Private dblA_coll_absorb As Double

' size of collector aperture
Private dblA_coll_aperture As Double

' size of collector gross
Private dblA_coll_gross As Double

' efficiency curve intercept
Private dblEtaZero As Double

' efficiency curve slope
Private dblK1 As Double

' efficiency curve curvature
Private dblK2 As Double

' collector efficiency factor
Private dblFPrime As Double

' total loss coefficient linear
Private dblU_LLin As Double

' pump mass flow rate
Private dblM_pump As Double

' pump electricity demand
Private dblP_pumpsolar As Double

' Collector heat capacitance
Private dblC_collector As Double

' transmission absorption product
Private dblTau_alpha As Double

' slope of the array
Private dblSlope As Double

' azimute of the array
Private dblAzimuth As Double

' // Arrays to store the model output variables
' temperature of solar thermal collector (including heat transfer fluid)
Private aTheta_collector(1 To 1440, 1 To 1) As Double

' useful heat delivered to hot water cylinder
Private aPhi_s(1 To 1440, 1 To 1) As Double

' collector system's electricity demand (due to pump)
Private aP_pumpsolar(1 To 1440, 1 To 1) As Double

' Incident solar power on collector (W)
Private aP_incident(1 To 1440, 1 To 1) As Double

' an array to store the control state of the collector (whether the pump is on or off)
Private aSolarThermalOnOff(1 To 1440, 1 To 1) As Double



' ===============================================================================================
' Class Properties

Public Property Get GetDailySumPhi_s() As Double
    GetDailySumPhi_s = WorksheetFunction.Sum(aPhi_s)
End Property

Public Property Get GetPhi_s(timestep As Integer) As Double
    GetPhi_s = aPhi_s(timestep, 1)
End Property

Public Property Get GetP_pumpsolar(timestep As Integer) As Double
    GetP_pumpsolar = aP_pumpsolar(timestep, 1)
End Property




' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseSolarThermal
'
' ===============================================================================================
Public Sub InitialiseSolarThermal(dwellingIndex As Integer, runNumber As Integer)
    Dim intOffset As Integer
    intOffset = 4
    ' // Get indexes
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    intSolarThermalIndex = Application.index(wsDwellings.Range("rSolarThermalIndex"), intOffset + intDwellingIndex).Value
    
    ' // Check if the dwelling has solar thermal
    If intSolarThermalIndex > 0 Then
        
        ' // Get system variables
        With wsSolarThermal
        
            dblN = .Cells(intOffset + intSolarThermalIndex, 3).Value
            dblA_coll_aperture = .Cells(intOffset + intSolarThermalIndex, 4).Value
            dblA_coll_absorb = .Cells(intOffset + intSolarThermalIndex, 5).Value
            dblA_coll_gross = .Cells(intOffset + intSolarThermalIndex, 6).Value
            dblEtaZero = .Cells(intOffset + intSolarThermalIndex, 7).Value
            dblK1 = .Cells(intOffset + intSolarThermalIndex, 8).Value
            dblK2 = .Cells(intOffset + intSolarThermalIndex, 9).Value
            dblFPrime = .Cells(intOffset + intSolarThermalIndex, 10).Value
            dblU_LLin = .Cells(intOffset + intSolarThermalIndex, 11).Value
            dblM_pump = .Cells(intOffset + intSolarThermalIndex, 12).Value
            dblP_pumpsolar = .Cells(intOffset + intSolarThermalIndex, 13).Value
            dblC_collector = .Cells(intOffset + intSolarThermalIndex, 14).Value
            dblTau_alpha = .Cells(intOffset + intSolarThermalIndex, 15).Value
            dblSlope = .Cells(intOffset + intSolarThermalIndex, 16).Value
            dblAzimuth = .Cells(intOffset + intSolarThermalIndex, 17).Value
            
        End With
        
    End If
    
    ' // Set initial temperature of the collector to that of external air
    aTheta_collector(1, 1) = aLocalClimate(intRunNumber).GetTheta_o(1)
End Sub




' ===============================================================================================
' ## CalculateSolarThermalOutput
'
' ===============================================================================================
Public Sub CalculateSolarThermalOutput(currentTimeStep As Integer)
    
    ' // Declare variables
    
    ' incident irradiance on collector
    Dim dblG_i As Double
    
    ' total loss coefficent of collector
    Dim dblU_L As Double
        
    ' temperature of outside air
    Dim dblTheta_o As Double
    
    ' temperature of water in hot water cylinder
    Dim dblTheta_cyl As Double
    
    ' temperature of collector
    Dim dblTheta_collector As Double
    
    ' change in temperature of the collector
    Dim dblDeltaTheta_collector As Double
    
    ' useful heat flow from collector
    Dim dblPhi_u As Double
    
    ' temperature difference between inlet and outlet for pump to start
    Dim dblTheta_control As Double
    
    ' max temperature for inlet for pump to switch off
    Dim dblTheta_max As Double
    
    ' time interval for time step
    Dim intTimeInterval As Integer
    
    'Intensity change
    Dim DeltaI As Double
    
    ' // determine the time interval for the time step (s)
    intTimeInterval = 60
    
    ' // Set the maximum temperature for switching off the panel
    dblTheta_max = 70
    
    ' // Set the control temperature difference
    dblTheta_control = 2
    
    ' // Check if dwelling has solar thermal
    If intSolarThermalIndex = 0 Then
        ' // If not, then there is no solar thermal output
        aPhi_s(currentTimeStep, 1) = 0
        aSolarThermalOnOff(currentTimeStep, 1) = 0
        aP_pumpsolar(currentTimeStep, 1) = 0
        aP_incident(currentTimeStep, 1) = 0
        
    ' // otherwise proceed to calculate solar thermal output
    Else
        ' // Get net irradiance incident on collector
        dblG_i = GetIncidentRadiation(currentTimeStep)
        
        ' // Calculate incident solar power on collector area
        aP_incident(currentTimeStep, 1) = dblA_coll_aperture * dblG_i * dblN
        
        ' // Get temperatures in previous timestep
        ' // (unless this is the first time step)
        If currentTimeStep = 1 Then
            dblTheta_cyl = aBuilding(intRunNumber).GetTheta_cyl(currentTimeStep)
            dblTheta_collector = aTheta_collector(currentTimeStep, 1)
            dblTheta_o = aLocalClimate(intRunNumber).GetTheta_o(currentTimeStep)
            DeltaI = 1
        Else
            dblTheta_cyl = aBuilding(intRunNumber).GetTheta_cyl(currentTimeStep - 1)
            dblTheta_collector = aTheta_collector(currentTimeStep - 1, 1)
            dblTheta_o = aLocalClimate(intRunNumber).GetTheta_o(currentTimeStep - 1)
            DeltaI = Abs((aP_incident(currentTimeStep, 1) / dblA_coll_aperture) - (aP_incident(currentTimeStep - 1, 1) / dblA_coll_aperture))
        End If
        
        ' // Given these temperatures, determine whether the pump should be switched on
        If ((dblTheta_collector - dblTheta_cyl) > dblTheta_control) And (dblTheta_cyl < dblTheta_max) Then
            
            aSolarThermalOnOff(currentTimeStep, 1) = 1
            
        Else
            aSolarThermalOnOff(currentTimeStep, 1) = 0
            
        End If
        
        ' // calculate the useful heat flow to the cylinder tank
            aPhi_s(currentTimeStep, 1) = dblN * aSolarThermalOnOff(currentTimeStep, 1) * dblM_pump _
                                    * SPECIFIC_HEAT_CAPACITY_WATER * (dblTheta_collector - dblTheta_cyl)
                                    
        '// calculate total loss coefficient of collector
        dblU_L = (dblK1 + dblK2 * (dblTheta_collector - dblTheta_o)) / dblFPrime
        
        ' // calculate the temperature of the collector
        dblDeltaTheta_collector = intTimeInterval / dblC_collector * _
                ( _
                dblTau_alpha * dblG_i * dblA_coll_absorb _
                - dblA_coll_aperture * dblU_L * (dblTheta_collector - dblTheta_o) _
                - (aPhi_s(currentTimeStep, 1) / dblN) _
                )
                
        aTheta_collector(currentTimeStep, 1) = dblTheta_collector + dblDeltaTheta_collector
        
        ' assign pump power
        aP_pumpsolar(currentTimeStep, 1) = dblP_pumpsolar * aSolarThermalOnOff(currentTimeStep, 1)
        
        
    End If
    
End Sub




' ===============================================================================================
' ## WriteSolarThermalOutput
'
' ===============================================================================================
Public Sub WriteSolarThermal(dwellingIndexRowOffset As Long)

    wsResultsDisaggregated.Range("AB7:AB1446").Offset(dwellingIndexRowOffset, 0) = aP_incident
    wsResultsDisaggregated.Range("AC7:AC1446").Offset(dwellingIndexRowOffset, 0) = aSolarThermalOnOff
    wsResultsDisaggregated.Range("AD7:AD1446").Offset(dwellingIndexRowOffset, 0) = aTheta_collector
    wsResultsDisaggregated.Range("AE7:AE1446").Offset(dwellingIndexRowOffset, 0) = aPhi_s
    
End Sub



' ===============================================================================================
' Code Module Functions
'
' ===============================================================================================
' ## GetIncidentRadiation
'
' ===============================================================================================
Private Function GetIncidentRadiation(currentTimeStep As Integer) As Double

    ' a variable to store the hour of the day
    Dim intHour As Integer
    
    ' a variable to store the minute of the hour
    Dim intMinute As Integer
    
    ' direct beam radiation on panel
    Dim dblG_direct As Double
    
    ' diffuse radiation on panel
    Dim dblG_diffuse As Double
    
    ' reflected radiation on panel
    Dim dblG_reflected As Double
    
    ' total clearsky radiation incident on array (W/m^2)
    Dim dG_iClearsky As Double
    
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
    
    Dim dblDayOfYear As Double
    
    Dim dblLongitude As Double
    
    Dim dblLatitude As Double
    
    Dim dblMeridian As Double
    
    Dim dblSkyDiffuseFactor As Double
    
    ' // Get longitude, latitude and meridian
    dblLongitude = wsMain.Range("rLongitude").Value
    dblLatitude = wsMain.Range("rLatitude").Value
    dblMeridian = wsMain.Range("rMeridian").Value

    ' // Determine the hour and minute
    intHour = WorksheetFunction.RoundDown(currentTimeStep / 60, 0)
    intMinute = currentTimeStep - intHour * 60
    ' // Calculate B
    dblB = 360 * (dblDayOfYear - 81) / 364
    
    ' // Calculate equation of time
    dblEquationOfTime = (9.87 * Sin(2 * dblB * PI / 180)) - (7.53 * Cos(dblB * PI / 180)) - (1.5 * Sin(dblB * PI / 180))
    
    ' // Calculate time correction factor
    dblTimeCorrectionFactor = (4 * (dblLongitude - dblMeridian)) + dblEquationOfTime
                                       
    ' // Calculate sky diffuse factor
    dblSkyDiffuseFactor = 0.095 + (0.04 * Sin(2 * PI * (dblDayOfYear - 100) / 365))
               
    ' // Update local standard time hour
    If ((dblDayOfYear >= DAY_SUMMER_TIME_STARTS And dblDayOfYear < DAY_SUMMER_TIME_END) _
    And wsMain.Shapes("objOverWriteData").ControlFormat.Value = 1) Then
        intLocalStandardTimeHour = intHour - 1
    Else
        intLocalStandardTimeHour = intHour
    End If
    
    ' // update local standard time minute
    intLocalStandardTimeMinute = intMinute
    
    ' // Calculate hours before solar noon
    dblHoursBeforeSolarNoon = 12 - (intLocalStandardTimeHour + (intLocalStandardTimeMinute / 60) + (dblTimeCorrectionFactor / 60))
                       
    ' // Calculate optical depth
    dblOpticalDepth = 0.174 + (0.035 * Sin(2 * PI * (dblDayOfYear - 100) / 365))
                
    ' // Calculate hour angle
    dblHourAngle = 15 * dblHoursBeforeSolarNoon
    
    ' // Calculate declination
    dblDeclination = 23.45 * Sin(2 * PI * (284 + dblDayOfYear) / 365.25)
    
    ' // Calculate solar altitude angle
    dblSolarAltitudeAngle = 180 / PI * Application.WorksheetFunction.Asin((Cos(dblLatitude * PI / 180) * Cos(dblDeclination * PI / 180) * Cos(dblHourAngle * PI / 180)) + (Sin(dblLatitude * PI / 180) * Sin(dblDeclination * PI / 180)))
    
    ' // Calculate azimuth of sun
    dblAzimuthOfSun = 180 / PI * Application.WorksheetFunction.Asin(Cos(dblDeclination * PI / 180) * Sin(dblHourAngle * PI / 180) / Cos(dblSolarAltitudeAngle * PI / 180))
    
    ' // Calculate azimuth angle test
    If Cos(dblHourAngle * PI / 180) >= (Tan(dblDeclination * PI / 180) / Tan(dblDeclination * PI / 180)) Then
        dblAzimuthAngleTest = 0
    Else
        dblAzimuthAngleTest = 1
    End If
    
    ' // Calculate adjusted azimuth of sun
    If (dblAzimuthAngleTest = 0) Then
    
        If (dblAzimuthOfSun > 90) Then
        
            dblAdjustedAzimuthOfSun = dblAzimuthOfSun - 90
            
        ElseIf (dblAzimuthOfSun < -90) Then
        
            dblAdjustedAzimuthOfSun = dblAzimuthOfSun + 90
            
        Else
        
            dblAdjustedAzimuthOfSun = dblAzimuthOfSun
            
        End If
        
    ElseIf (dblAzimuthOfSun > 0 And dblAzimuthOfSun < 90) Then
    
        dblAdjustedAzimuthOfSun = 180 - dblAzimuthOfSun
        
    ElseIf (dblAzimuthOfSun > -90 And dblAzimuthOfSun < 0) Then
    
        dblAdjustedAzimuthOfSun = -180 - dblAzimuthOfSun
        
    Else
    
        dblAdjustedAzimuthOfSun = dblAzimuthOfSun
        
    End If
        
    ' // Calculate solar incident angle on panel
    dblSolarIncidentAngle = Cos((Cos(dblSolarAltitudeAngle * PI / 180) * Cos(dblAdjustedAzimuthOfSun * PI / 180 - dblAzimuth * PI / 180) * Sin(dblSlope * PI / 180)) + (Sin(dblSolarAltitudeAngle * PI / 180) * Cos(dblSlope * PI / 180)))
    
    ' // Get clearsky beam radiation at surface (plane tracking the sun)
    dblG_oClearsky = aLocalClimate(intRunNumber).GetG_oClearsky(currentTimeStep)
                
    ' // Calculate direct beam radiation on panel
    If Abs(dblSolarIncidentAngle) > 90 Then
        
        dblG_direct = 0
        
    Else
    
        dblG_direct = dblG_oClearsky * Cos(dblSolarIncidentAngle * PI / 180)
        
    End If
    
    ' // Calculate diffuse radiation on panel
    dblG_diffuse = dblSkyDiffuseFactor * dblG_oClearsky * ((1 + Cos(dblSlope * PI / 180)) / 2)
    
    ' // Calculate reflected radiation on panel
    dblG_reflected = GROUND_REFLECTANCE * dblG_oClearsky * (Sin(dblSolarAltitudeAngle * PI / 180) + dblSkyDiffuseFactor) * ((1 - Cos(dblSlope * PI / 180)) / 2)
                
    ' // Total clearsky incident radiation on array is the sum of direct diffuse and reflected radiation
    dG_iClearsky = dblG_direct + dblG_diffuse + dblG_reflected
    
    ' // Get clearness index for this time step
    dblClearnessIndex = aLocalClimate(intRunNumber).GetClearnessIndex(currentTimeStep)
                
    ' // Calculate net radiation on collector and return value
    GetIncidentRadiation = dG_iClearsky * dblClearnessIndex
    
    
End Function



