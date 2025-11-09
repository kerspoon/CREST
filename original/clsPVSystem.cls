VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsPVSystem"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## PV system and irradiance model
' ===============================================================================================

Option Explicit

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

Private intPVSystemIndex As Integer
Private intRunNumber As Integer
 
' // PV system variables
' size of PV array
Private dblA_array As Double

' efficiency of the PV system
Private dblEta_pv As Double

' slope of the array
Private dblSlope As Double

' azimute of the array
Private dblAzimuth As Double

' // Arrays to store the model output variables
' net radiation incident on PV array (W/m^2)
Private aG_i(1 To 1440, 1 To 1) As Double

' electricity output of PV system (W)
Private aP_pv(1 To 1440, 1 To 1) As Double

' net dwelling electricity demand (W)
Private aP_net(1 To 1440, 1 To 1) As Double

' dwelling self-consumption (W)
Private aP_self(1 To 1440, 1 To 1) As Double


' ===============================================================================================
' Class Properties

Public Property Get GetDailySumPvOutput() As Double
    GetDailySumPvOutput = WorksheetFunction.Sum(aP_pv)
End Property

Public Property Get GetDailySumP_net() As Double
    GetDailySumP_net = WorksheetFunction.Sum(aP_net)
End Property

Public Property Get GetDailySumP_self() As Double
    GetDailySumP_self = WorksheetFunction.Sum(aP_self)
End Property


' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialisePVSystem
'
' ===============================================================================================
Public Sub InitialisePVSystem(dwellingIndex As Integer, runNumber As Integer)
    Dim intOffset As Integer
    Dim blnPVOption As Boolean
    intOffset = 4
    ' // Get indexes
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    
    If wsMain.Shapes("objPVOption").ControlFormat.Value = 1 Then
        intPVSystemIndex = Application.index(wsDwellings.Range("rPvSystemIndex"), intOffset + intDwellingIndex).Value
    Else
        intPVSystemIndex = 0
    End If
    
    ' // Check if the dwelling has PV
    If intPVSystemIndex > 0 Then
        
        ' // Get PV system variables
        With wsPVSystems
            dblA_array = Application.index(.Range("rA_array"), intOffset + intPVSystemIndex).Value
            dblEta_pv = Application.index(.Range("rEta_pv"), intOffset + intPVSystemIndex).Value
            dblAzimuth = Application.index(.Range("rAzimuth"), intOffset + intPVSystemIndex).Value
            dblSlope = Application.index(.Range("rSlope"), intOffset + intPVSystemIndex).Value
        End With
        
    End If
    
End Sub




' ===============================================================================================
' ## CalculatePVOutput
'
' ===============================================================================================
Public Sub CalculatePVOutput()
    
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
    
    Dim strDate As String
    
    Dim intDayOfMonth As Integer
        
    Dim intMonthOfYear As Integer
    
    Dim dblDayOfYear As Double
    
    Dim dblLongitude As Double
    
    Dim dblLatitude As Double
    
    Dim dblMeridian As Double
    
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
    
    ' // Check if the dwelling has PV
    If intPVSystemIndex = 0 Then
        For intCount = 1 To 1440
            aP_pv(intCount, 1) = 0
            
        Next intCount
    Else
                   
        ' // Calculate B
        dblB = 360 * (dblDayOfYear - 81) / 364
        
        ' // Calculate equation of time
        dblEquationOfTime = (9.87 * Sin(2 * dblB * PI / 180)) - (7.53 * Cos(dblB * PI / 180)) - (1.5 * Sin(dblB * PI / 180))
        
        ' // Calculate time correction factor
        dblTimeCorrectionFactor = (4 * (dblLongitude - dblMeridian)) + dblEquationOfTime
               
        ' // Calculate sky diffuse factor
        dblSkyDiffuseFactor = 0.095 + (0.04 * Sin(2 * PI * (dblDayOfYear - 100) / 365))
                        
        ' // Start the count variable
        intCount = 1
        
        ' // Loop through the hours of the day
        For intHour = 0 To 23
        
            ' // Update local standard time hour
            If ((dblDayOfYear >= DAY_SUMMER_TIME_STARTS And dblDayOfYear < DAY_SUMMER_TIME_END) _
            And wsMain.Shapes("objDaylightSaving").ControlFormat.Value = 1) Then
                intLocalStandardTimeHour = intHour - 1
            Else
                intLocalStandardTimeHour = intHour
            End If
            
            ' // Loop through the minutes of the day
            For intMinute = 1 To 60
            
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
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 10) = dblSolarAltitudeAngle
                
                ' // Calculate azimuth of sun
                dblAzimuthOfSun = 180 / PI * Application.WorksheetFunction.Asin(Cos(dblDeclination * PI / 180) * Sin(dblHourAngle * PI / 180) / Cos(dblSolarAltitudeAngle * PI / 180))
                
               ' // Check whether the acute angle is the correct azimuth:
                If Cos(dblHourAngle * PI / 180) >= (Tan(dblDeclination * PI / 180) / Tan(dblLatitude * PI / 180)) Then
                    '//Azimuth is acute. Output of Asin should be an acute angle anyway, so no action needed.
                    dblAdjustedAzimuthOfSun = dblAzimuthOfSun
                    Dim tim As Double
                    tim = 1
                Else
                    '//Azimuth is obtuse.
                    If (dblAzimuthOfSun > 0) Then
                        dblAdjustedAzimuthOfSun = 180 - dblAzimuthOfSun
                    ElseIf (dblAzimuthOfSun < 0) Then
                        dblAdjustedAzimuthOfSun = -180 - dblAzimuthOfSun
                    End If
                End If
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 11) = dblAdjustedAzimuthOfSun
                
                ' // Calculate solar incident angle on panel
                Dim dotProduct As Double
                dotProduct = (Cos(dblSolarAltitudeAngle * PI / 180) * Cos(dblAdjustedAzimuthOfSun * PI / 180 - dblAzimuth * PI / 180) * Sin(dblSlope * PI / 180)) + (Sin(dblSolarAltitudeAngle * PI / 180) * Cos(dblSlope * PI / 180))
                
                dblSolarIncidentAngle = (180 / PI) * WorksheetFunction.Acos(dotProduct)
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 9) = dblSolarIncidentAngle
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 12) = dotProduct
                
                ' // Get clearsky beam radiation at surface (plane tracking the sun)
                dblG_oClearsky = aLocalClimate(intRunNumber).GetG_oClearsky(intCount)
                            
                ' // Calculate direct beam radiation on panel
                If Abs(dblSolarIncidentAngle) > 90 Then
                    dblG_direct = 0
                Else
                    dblG_direct = dblG_oClearsky * Cos(dblSolarIncidentAngle * PI / 180)
                End If
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 2) = dblG_direct
                
                ' // Calculate diffuse radiation on panel
                dblG_diffuse = dblSkyDiffuseFactor * dblG_oClearsky * ((1 + Cos(dblSlope * PI / 180)) / 2)
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 3) = dblG_diffuse
                
                ' // Calculate reflected radiation on panel
                dblG_reflected = GROUND_REFLECTANCE * dblG_oClearsky * (Sin(dblSolarAltitudeAngle * PI / 180) + dblSkyDiffuseFactor) * ((1 - Cos(dblSlope * PI / 180)) / 2)
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 4) = dblG_reflected
                            
                ' // Total clearsky incident radiation on array is the sum of direct diffuse and reflected radiation
                dG_iClearsky = dblG_direct + dblG_diffuse + dblG_reflected
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 5) = dG_iClearsky
                                
                ' // Get clearness index for this time step
                dblClearnessIndex = aLocalClimate(intRunNumber).GetClearnessIndex(intCount)
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 6) = dblClearnessIndex
                                
                ' // Calculate net radiation on panel
                aG_i(intCount, 1) = dG_iClearsky * dblClearnessIndex
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 7) = dG_iClearsky * dblClearnessIndex
                
                ' // Calculate the PV output
                aP_pv(intCount, 1) = aG_i(intCount, 1) * dblA_array * dblEta_pv
                ThisWorkbook.Worksheets("Irradiance").Cells(intCount + 3, 8) = aG_i(intCount, 1) * dblA_array * dblEta_pv
                 
                ' // Iterate the count variable
                intCount = intCount + 1
                
            Next intMinute
        Next intHour
    End If
End Sub




' ===============================================================================================
' ## CalculateNetDemand
'
' ===============================================================================================
Public Sub CalculateNetDemand()
    Dim intMinute As Integer
    
    Dim dblTotalApplianceDemand As Double
    
    Dim dblTotalLightingDemand As Double
    
    For intMinute = 1 To 1440
    
        ' // Calculate the dwelling net electricity demand
        dblTotalApplianceDemand = aAppliances(intRunNumber).GetTotalApplianceDemand(intMinute)
        dblTotalLightingDemand = aLighting(intRunNumber).GetTotalLightingDemand(intMinute)
        
        aP_net(intMinute, 1) = dblTotalApplianceDemand + dblTotalLightingDemand - aP_pv(intMinute, 1)
    
    Next intMinute
    
End Sub




' ===============================================================================================
' ## CalculateNetDemand
'
' ===============================================================================================
Public Sub CalculateSelfConsumption()
    Dim intMinute As Integer
    
    For intMinute = 1 To 1440
        aP_self(intMinute, 1) = WorksheetFunction.Min _
        ( _
        aLighting(intRunNumber).GetTotalLightingDemand(intMinute) + aAppliances(intRunNumber).GetTotalApplianceDemand(intMinute), _
        aP_pv(intMinute, 1) _
        )
        
    Next intMinute
End Sub


' ===============================================================================================
' ## WritePVSystem
'
' ===============================================================================================
Public Sub WritePVSystem(dwellingIndexRowOffset As Long)
    
    With wsResultsDisaggregated
        .Range("V7:V1446").Offset(dwellingIndexRowOffset, 0) = aG_i
        .Range("W7:W1446").Offset(dwellingIndexRowOffset, 0) = aP_pv
        .Range("X7:X1446").Offset(dwellingIndexRowOffset, 0) = aP_net
        .Range("AF7:AF1446").Offset(dwellingIndexRowOffset, 0) = aP_self
    End With
End Sub
