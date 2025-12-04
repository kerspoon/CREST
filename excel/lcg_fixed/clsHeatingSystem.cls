Attribute VB_Name = "clsHeatingSystem"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
' ===============================================================================================
' ## Heating system model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

Private intDwellingIndex As Integer
Private intRunNumber As Integer

Private intHeatingSystemIndex As Integer

' Type of heating unit (e.g. boiler)
Private strHeatingType As String

' Type of heating system (e.g. regular, combi)
Private intHeatingSystemType As Integer

' Type of fuel
Private strFuelType As String

' Nominal fuel flow rate
Private dblFuelFlowRate As Double

' Heat output of unit
Private dblPhi_h As Double

' Standby power of unit
Private dblP_standby As Double

' Pump power of unit
Private dblP_pump As Double

' thermal efficiency of unit
Private dblEta_h As Double

' time-series array of heating system output (W)
Private aPhi_h(1 To 1440, 1 To 1) As Double

' array for heat to hot water
Private aPhi_hWater(1 To 1440, 1 To 1) As Double

' array for heat to space
Private aPhi_hSpace(1 To 1440, 1 To 1) As Double

' array for fuel flow rate
Private aM_fuel(1 To 1440, 1 To 1) As Double

' array for electricity used for heating
Private aHeatingElectricity(1 To 1440, 1 To 1) As Double

' array for unit's electricity demand
Private aP_h(1 To 1440, 1 To 1) As Double




' ===============================================================================================
' Class Properties

Public Property Get GetPhi_h(currentTimeStep As Integer)
    GetPhi_h = aPhi_h(currentTimeStep, 1)
End Property

Public Property Get GetHeaterOutput(currentTimeStep As Integer)
    GetHeaterOutput = aPhi_h(currentTimeStep, 1)
End Property

Public Property Get GetHeatToSpace(currentTimeStep As Integer)
    GetHeatToSpace = aPhi_hSpace(currentTimeStep, 1)
End Property

Public Property Get GetHeatToHotWater(currentTimeStep As Integer)
    GetHeatToHotWater = aPhi_hWater(currentTimeStep, 1)
End Property

Public Property Get GetHeatingSystemPowerDemand(currentTimeStep As Integer)
    GetHeatingSystemPowerDemand = aP_h(currentTimeStep, 1) + aHeatingElectricity(currentTimeStep, 1)
End Property

Public Property Get GetHeatingSystemType() As Double
    GetHeatingSystemType = intHeatingSystemType
End Property

Public Property Get GetDailySumThermalEnergySpace() As Double
    GetDailySumThermalEnergySpace = WorksheetFunction.Sum(aPhi_hSpace)
End Property

Public Property Get GetDailySumThermalEnergyWater() As Double
    GetDailySumThermalEnergyWater = WorksheetFunction.Sum(aPhi_hWater)
End Property

Public Property Get GetDailySumFuelFlow() As Double
    GetDailySumFuelFlow = WorksheetFunction.Sum(aM_fuel)
End Property

Public Property Get GetDailySumHeatingElectricity() As Double
    GetDailySumHeatingElectricity = WorksheetFunction.Sum(aHeatingElectricity)
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseHeatingSystem
'
' ===============================================================================================
Public Sub InitialiseHeatingSystem(dwellingIndex As Integer, runNumber As Integer)
    
    Dim intMinute As Integer
    Dim intOffset As Integer
    
    intOffset = 4
    
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    intHeatingSystemIndex = Application.index(wsDwellings.Range("rPrimaryHeatingIndex"), intOffset + intDwellingIndex).Value
    
    With wsPrimaryHeatingSystems
    
        strHeatingType = Application.index(.Range("rHeatingType"), intOffset + intHeatingSystemIndex).Value
        intHeatingSystemType = Application.index(.Range("rHeatingSystemType"), intOffset + intHeatingSystemIndex).Value
        strFuelType = Application.index(.Range("rFuelType"), intOffset + intHeatingSystemIndex).Value
        dblFuelFlowRate = Application.index(.Range("rFuelFlowRate"), intOffset + intHeatingSystemIndex).Value
        dblPhi_h = Application.index(.Range("rPhi_h"), intOffset + intHeatingSystemIndex).Value
        dblP_standby = Application.index(.Range("rP_standby"), intOffset + intHeatingSystemIndex).Value
        dblP_pump = Application.index(.Range("rP_pump"), intOffset + intHeatingSystemIndex).Value
        dblEta_h = Application.index(.Range("rEta_h"), intOffset + intHeatingSystemIndex).Value
            
    End With
        
End Sub

' ===============================================================================================
' ## CalculateHeatOutput (and Cooling output)
'
' ===============================================================================================
Public Sub CalculateHeatOutput(currentTimeStep As Integer)
    
    ' Boolean to store whether the heating system needs to be switched on in this time step
    Dim blnHeaterOnOff As Boolean
    
    ' Is heating required for the hot water
    Dim blnHeatWaterOnOff As Boolean
    
    ' boolean to determine if space heating is required
    Dim blnSpaceHeatingOnOff As Boolean
    
    ' heat delivered to heat hot water
    Dim dblPhi_hWater As Double
    
    ' heat or cooling delivered to heat, or cool space
    Dim dblPhi_hSpace As Double
    
    ' total heat output of heating system
    Dim dblPhi_hTotal As Double
    
    ' target heat output for delivery to hot water
    Dim dblPhi_hWaterTarget As Double
    
    ' target outputs for delivery of heating or cooling
    Dim dblPhi_hSpaceTarget As Double
    
'wsMain.Range("M15") = " Started this bit "
    ' // Get the control signals from the heating controller
    blnHeaterOnOff = aHeatingControls(intRunNumber).GetHeaterOnOff(currentTimeStep)
    blnHeatWaterOnOff = aHeatingControls(intRunNumber).GetHeatWaterOnOff(currentTimeStep)
    blnSpaceHeatingOnOff = aHeatingControls(intRunNumber).GetSpaceThermostatState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetSpaceTimerState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetEmitterThermostatState(currentTimeStep)
                                
'wsMain.Range("M15") = " Finished this bit blnHeaterOnOff: " + CStr(blnHeaterOnOff) + " blnHeatWaterOnOff: " + CStr(blnHeatWaterOnOff) + " blnSpaceHeatingOnOff: " + CStr(blnSpaceHeatingOnOff)
    ' // First set the default variables for the heating system
    aPhi_hWater(currentTimeStep, 1) = 0
    aPhi_hSpace(currentTimeStep, 1) = 0
    aPhi_h(currentTimeStep, 1) = 0
    aM_fuel(currentTimeStep, 1) = 0
    
    ' // the heating system pump will operate if space heating timer and thermostat states are on
    ' // and even if emitter return thermostat state is off
'wsMain.Range("M16") = " Started this bit"
    aP_h(currentTimeStep, 1) = IIf(aHeatingControls(intRunNumber).GetSpaceThermostatState(currentTimeStep) _
                            * aHeatingControls(intRunNumber).GetSpaceTimerState(currentTimeStep) = 1, _
                            dblP_pump, _
                            dblP_standby)
'wsMain.Range("M16") = " Finished this bit aP_h: " + CStr(aP_h(currentTimeStep, 1))
    ' // does the heating system need to be switched on?
    If blnHeaterOnOff Then
        
        ' // if the hot water is needed, get the target heat output required for the hot water
'wsMain.Range("M17") = " Started this bit "
        If blnHeatWaterOnOff Then
        
            dblPhi_hWaterTarget = aBuilding(intRunNumber).GetPhi_hWater(currentTimeStep)
'wsMain.Range("M17") = " Finished this bit dblPhi_hWaterTarget: " + CStr(dblPhi_hWaterTarget)
            ' // assign heat to hot water, bound by max and min values
'wsMain.Range("M18") = " Started this bit "
            dblPhi_hWater = WorksheetFunction.Max(0, (WorksheetFunction.Min(dblPhi_h, dblPhi_hWaterTarget)))
            aPhi_hWater(currentTimeStep, 1) = dblPhi_hWater
'wsMain.Range("M18") = " Finished this bit dblPhi_hWater: " + CStr(dblPhi_hWater)
            ' // if space heating is also required, then assign remaining required capacity
'wsMain.Range("M19") = "Started this bit currentTimeStep " + CStr(currentTimeStep)
            If blnSpaceHeatingOnOff Then
                dblPhi_hSpaceTarget = aBuilding(intRunNumber).GetPhi_hSpace(currentTimeStep)
'wsMain.Range("M19") = " Finished this bit dblPhi_hSpaceTarget: " + CStr(dblPhi_hSpaceTarget)
                ' // assign remaining required capacity to space heating
'wsMain.Range("M20") = " Starte this bit"
                dblPhi_hSpace = WorksheetFunction.Max(0, (WorksheetFunction.Min(dblPhi_h - dblPhi_hWater, dblPhi_hSpaceTarget)))
                aPhi_hSpace(currentTimeStep, 1) = dblPhi_hSpace
'wsMain.Range("M20") = " Finished this bit dblPhi_hSpace: " + CStr(dblPhi_hSpace)
            End If
            
        ' // otherwise heat is required only for space heating
        Else
'wsMain.Range("M21") = " Started this bit"
'Stop
            ' // get the target heat output for space heating
            dblPhi_hSpaceTarget = aBuilding(intRunNumber).GetPhi_hSpace(currentTimeStep)
            
            ' // assign heat to space heating, bound by max and min values
            dblPhi_hSpace = WorksheetFunction.Max(0, (WorksheetFunction.Min(dblPhi_h, dblPhi_hSpaceTarget)))
            aPhi_hSpace(currentTimeStep, 1) = dblPhi_hSpace
'wsMain.Range("M21") = " Finished this bit"
            
        End If
'wsMain.Range("M22") = " Started this bit"
        ' // assign remaining heating system variables
        dblPhi_hTotal = aPhi_hSpace(currentTimeStep, 1) + aPhi_hWater(currentTimeStep, 1)
        aPhi_h(currentTimeStep, 1) = dblPhi_hTotal
'wsMain.Range("M22") = " Finished this bit dblPhi_hTotal: " + CStr(dblPhi_hTotal)
        aP_h(currentTimeStep, 1) = dblP_pump
'wsMain.Range("M23") = " dblP_pump: " + CStr(dblP_pump) + " dblPhi_hTotal: " + CStr(dblPhi_hTotal) + " dblPhi_h: " + CStr(dblPhi_h)
        If dblPhi_h > 0 Then
            If intHeatingSystemIndex <= 3 Then
                aM_fuel(currentTimeStep, 1) = dblFuelFlowRate * dblPhi_hTotal / dblPhi_h
            Else
                aHeatingElectricity(currentTimeStep, 1) = dblFuelFlowRate * 1000 * dblPhi_hTotal / dblPhi_h
            End If
        End If
'wsMain.Range("M23") = " Got this far aM_fuel: " + CStr(aM_fuel(currentTimeStep, 1))
    End If
'wsMain.Range("M24") = " Got this far"
          
End Sub




' ===============================================================================================
' ## WriteHeatingSystem
'
' ===============================================================================================
Public Sub WriteHeatingSystem(dwellingIndexRowOffset As Long)
    wsResultsDisaggregated.Range("L7:L1446").Offset(dwellingIndexRowOffset, 0) = aPhi_h
    wsResultsDisaggregated.Range("Y7:Y1446").Offset(dwellingIndexRowOffset, 0) = aPhi_hSpace
    wsResultsDisaggregated.Range("Z7:Z1446").Offset(dwellingIndexRowOffset, 0) = aPhi_hWater
    wsResultsDisaggregated.Range("AA7:AA1446").Offset(dwellingIndexRowOffset, 0) = aM_fuel
    wsResultsDisaggregated.Range("AN7:AN1446").Offset(dwellingIndexRowOffset, 0) = aHeatingElectricity
End Sub
