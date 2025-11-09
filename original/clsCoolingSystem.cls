VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "clsCoolingSystem"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
' ===============================================================================================
' ## Cooling system model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer
Private intRunNumber As Integer

Private intCoolingSystemIndex As Integer

' Type of heating unit (e.g. boiler)
Private strCoolingType As String

' Type of heating system (e.g. regular, combi)
Private intCoolingSystemType As Integer

' Maximum Heat output capability of unit
Private dblPhi_hCool As Double

' Standby power of unit
Private dblP_standbycool As Double

' Pump power of unit
Private dblP_pumpcool As Double

' thermal efficiency of unit
Private dblEta_hcool As Double

' array for cooling = heat absorbed from indoor space
Private aPhi_hCooling(1 To 1440, 1 To 1) As Double

' array for cooling unit's electricity demand
Private aPhi_Cooling(1 To 1440, 1 To 1) As Double
'


' ===============================================================================================
' Class Properties

Public Property Get GetCoolingToSpace(currentTimeStep As Integer)
    GetCoolingToSpace = aPhi_hCooling(currentTimeStep, 1)
End Property

Public Property Get GetCoolingSystemPowerDemand(currentTimeStep As Integer)
    GetCoolingSystemPowerDemand = aPhi_Cooling(currentTimeStep, 1)
End Property

Public Property Get GetCoolingSystemType() As Double
    GetCoolingSystemType = intCoolingSystemType
End Property

Public Property Get GetDailySumCoolingEnergy() As Double
    GetDailySumCoolingEnergy = WorksheetFunction.Sum(aPhi_hCooling)
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseCoolingSystem
'
' ===============================================================================================
Public Sub InitialiseCoolingSystem(dwellingIndex As Integer, runNumber As Integer)
    
    Dim intMinute As Integer
    Dim intOffset As Integer
    
    intOffset = 4
    
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    intCoolingSystemIndex = Application.index(wsDwellings.Range("rCoolingIndex"), intOffset + intDwellingIndex).Value
    
    With wsCoolingSystems
        ' Cooling system properties
        strCoolingType = Application.index(.Range("rCoolingType"), intOffset + intCoolingSystemIndex).Value
        intCoolingSystemType = Application.index(.Range("rCoolingSystemType"), intOffset + intCoolingSystemIndex).Value
        dblPhi_hCool = Application.index(.Range("rPhi_hcool"), intOffset + intCoolingSystemIndex).Value
        dblP_standbycool = Application.index(.Range("rP_standbycool"), intOffset + intCoolingSystemIndex).Value
        dblP_pumpcool = Application.index(.Range("rP_pumpcool"), intOffset + intCoolingSystemIndex).Value
        dblEta_hcool = Application.index(.Range("rEta_hcool"), intOffset + intCoolingSystemIndex).Value
            
    End With
        
End Sub



' ===============================================================================================
' ## CalculateCoolingOutput
'
' ===============================================================================================
Public Sub CalculateCoolingOutput(currentTimeStep As Integer)
    
    ' boolean to determine if space cooling is required
    Dim blnSpaceCoolingOnOff As Boolean
    
    ' heat or cooling delivered to heat, or cool space
    Dim dblPhi_hCooling As Double
    
    ' target outputs for delivery of heating or cooling
    Dim dblPhi_hCoolingTarget As Double
    
'wsMain.Range("M15") = " Started this bit "
    ' // Get the control signals from the heating controller
    blnSpaceCoolingOnOff = aHeatingControls(intRunNumber).GetSpaceCoolingThermostatState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetSpaceCoolingTimerState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetCoolerEmitterState(currentTimeStep)
                            
'wsMain.Range("M15") = " Finished this bit blnHeaterOnOff: " + CStr(blnHeaterOnOff) + " blnHeatWaterOnOff: " + CStr(blnHeatWaterOnOff) + " blnSpaceHeatingOnOff: " + CStr(blnSpaceHeatingOnOff)
    ' // First set the default variables for the heating system
    aPhi_hCooling(currentTimeStep, 1) = 0
    
'wsMain.Range("M24") = " Got this far"
    If blnSpaceCoolingOnOff Then
    
        dblPhi_hCoolingTarget = aBuilding(intRunNumber).GetPhi_hCooling(currentTimeStep)
        
        dblPhi_hCooling = WorksheetFunction.Min(0, (WorksheetFunction.Max(dblPhi_hCool, dblPhi_hCoolingTarget)))
                aPhi_hCooling(currentTimeStep, 1) = dblPhi_hCooling
                
        aPhi_Cooling(currentTimeStep, 1) = IIf(aHeatingControls(intRunNumber).GetSpaceCoolingThermostatState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetSpaceCoolingTimerState(currentTimeStep) = 1, _
                            dblP_pumpcool, _
                            dblP_standbycool)
                
        'wsMain.Range("M22") = " dblPhi_hCooling: " + CStr(dblPhi_hCooling)
    End If
      
End Sub




' ===============================================================================================
' ## WriteCoolingSystem
'
' ===============================================================================================
Public Sub WriteCoolingSystem(dwellingIndexRowOffset As Long)
    wsResultsDisaggregated.Range("AI7:AI1446").Offset(dwellingIndexRowOffset, 0) = aPhi_hCooling
    wsResultsDisaggregated.Range("AM7:AM1446").Offset(dwellingIndexRowOffset, 0) = aPhi_Cooling
End Sub


