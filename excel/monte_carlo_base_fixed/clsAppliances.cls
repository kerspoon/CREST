Attribute VB_Name = "clsAppliances"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
' ===============================================================================================
' ## Domestic Appliance Model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

' dwelling index number
Private intDwellingIndex As Integer
Private intRunNumber As Integer

' number of residents for the dwelling
Private intResidents As Integer

' array to store the appliance demands
Private aSimulationArray(1 To 1442, 1 To 31) As Variant

' dwelling's appliance configuration
Private aApplianceConfiguration(1 To 31, 1 To 1) As Boolean

' dwelling's active occupancy
Private aActiveOccupancy() As Integer

' total appliance demand for the dwelling
Private aTotalApplianceDemand(1 To 1440, 1 To 1) As Double

' casual thermal gains from dwelling's appliances
Private aApplianceThermalGains(1 To 1440, 1 To 1) As Double

' // declare the appliance description variables
Private strApplianceType As String
Private intMeanCycleLength As Integer
Private intCyclesPerYear As Integer
Private intStandbyPower As Integer
Private intRatedPower As Integer
Private dblProbSwitchOn As Double
Private dblOwnership As Double
Private intTargetAveragekWhYear As Integer
Private strUseProfile As String
Private intRestartDelay As Integer
Private blnHasAppliance As Boolean

' // declare timing counter and current power use variables
Private intCycleTimeLeft As Integer
Private intRestartDelayTimeLeft As Integer
Private intPower As Integer




' ===============================================================================================
' Class Properties

' // total appliance demand for a given time step
Public Property Get GetTotalApplianceDemand(timestep As Integer) As Double
    GetTotalApplianceDemand = aTotalApplianceDemand(timestep, 1)
End Property

' // Get the objects casual thermal gains property
Public Property Get ThermalGains() As Double()
    ThermalGains = aApplianceThermalGains()
End Property

' // Get the thermal gains for a specific time step
Public Property Get GetPhi_cAppliances(timestep As Integer)
    GetPhi_cAppliances = aApplianceThermalGains(timestep, 1)
End Property

Public Property Get GetDailySumApplianceDemand() As Double
    GetDailySumApplianceDemand = WorksheetFunction.Sum(aTotalApplianceDemand)
End Property




' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseAppliances
' Initialise the appliance object for this dwelling
' ===============================================================================================
Public Sub InitialiseAppliances(dwellingIndex As Integer, runNumber As Integer)
    
    ' // Get the dwelling index
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    
    ' // Get the active occupancy array for this dwelling
    aActiveOccupancy = aOccupancy(intRunNumber).GetActiveOccupancy
    
    ' // Configure appliances in the dwelling
    
    ' // Declare variables
    Dim i As Integer
    Dim intOffset As Integer
    Dim dblRan As Double
    Dim dblProportion As Double
    
    ' // Vertical offset
    intOffset = 12
    
    ' // For each appliance
    For i = 1 To 31
    
        ' // Get a random number
        dblRan = Rnd()
        
        ' // Get the proportion of houses with this appliance
        dblProportion = wsAppliancesAndWaterFixtures.Range("rApplianceProportion").Cells(i, 1).Value
        
        ' // Determine if this simulated dwelling has this appliance
        aApplianceConfiguration(i, 1) = IIf(dblRan < dblProportion, True, False)
 
    Next i
    
End Sub



' ===============================================================================================
' ## RunApplianceSimulation
' Run the appliance model
' ===============================================================================================
Public Sub RunApplianceSimulation()
    ' // Declare variables
    Dim intMonth As Integer
    Dim dblRan As Double
    Dim dblActivityProbability As Double
    Dim intTenMinuteCount As Integer
    Dim intAppliance As Integer
    Dim blnWeekend As Boolean
    Dim intMinute As Integer
    Dim intActiveOccupants As Integer
    Dim strKey As String
    Dim strCell As String
    Dim intApplianceSourceCellOffsetY As Integer
    Dim intDayOfYear As Integer
    
    ' // Define the cell count offsets
    intApplianceSourceCellOffsetY = 8
        
    ' // For each of the appliances
    For intAppliance = 1 To 31
    
        ' // Initialisation
        intCycleTimeLeft = 0
        intRestartDelayTimeLeft = 0
        ' // Get the appliance details
        With wsAppliancesAndWaterFixtures
            strApplianceType = .Range("E" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            intMeanCycleLength = .Range("R" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            intCyclesPerYear = .Range("X" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
    
            intStandbyPower = .Range("T" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            intRatedPower = .Range("P" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            dblProbSwitchOn = .Range("AD" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            dblOwnership = .Range("F" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
    'wsMain.Range("M18") = "Appliance number " + CStr(intAppliance) + " Got this far"
            intTargetAveragekWhYear = .Range("V" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
    'wsMain.Range("M19") = "Appliance number " + CStr(intAppliance) + " Got this far"
    'NB All parameters that get turned into integers must have values of 32768 or smaller
            strUseProfile = .Range("G" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
            intRestartDelay = .Range("S" + CStr(intAppliance + intApplianceSourceCellOffsetY)).Value
        End With
    
        blnHasAppliance = aApplianceConfiguration(intAppliance, 1)
        ' // Write the appliance type into the target sheet header
        aSimulationArray(1, intAppliance) = strApplianceType
        ' // Write the units
        aSimulationArray(2, intAppliance) = "(W)"
        ' // Check if this appliance is assigned to this dwelling
        If (Not blnHasAppliance) Then
            ' //This appliance is not applicable, so write zeros to the power demand
            Dim intCount As Integer
            For intCount = 3 To 1442
                aSimulationArray(intCount, intAppliance) = 0
            Next intCount
            
        Else
                 
            ' // Randomly delay the start of appliances that have a restart delay (e.g. cold appliances with more regular intervals)
            intRestartDelayTimeLeft = Rnd() * intRestartDelay * 2 ' // Weighting is 2 just to provide some diversity
            ' // Make the rated power variable over a normal distribution to provide some variation
            intRatedPower = GetMonteCarloNormalDistGuess(Val(intRatedPower), intRatedPower / 10)
    
            ' // Get the weekday or weekend flag
            blnWeekend = IIf(wsMain.Range("rDayType").Value = "wd", False, True)
             
            ' // Loop through each minute of the day
            intMinute = 1
            Do While (intMinute <= 1440)
                
                ' // Set the default (standby) power demand at this time step
                intPower = intStandbyPower
            
                ' // Get the ten minute period count
                intTenMinuteCount = ((intMinute - 1) \ 10)
            
                ' // Get the number of current active occupants for this minute
                intActiveOccupants = aActiveOccupancy(intTenMinuteCount, 0)
    
                ' // Now generate a key to get the activity statistics
                strKey = IIf(blnWeekend, "1", "0") + "_" + CStr(intActiveOccupants) + "_" + strUseProfile
    
                ' // If this appliance is off having completed a cycle (ie. a restart delay)
                If (intCycleTimeLeft <= 0) And (intRestartDelayTimeLeft > 0) Then
    
                    ' // Decrement the cycle time left
                    intRestartDelayTimeLeft = intRestartDelayTimeLeft - 1
                           
                ' // Else if this appliance is off
                ElseIf intCycleTimeLeft <= 0 Then
    
                    ' // There must be active occupants, or the profile must not depend on occupancy for a start event to occur
                    If (intActiveOccupants > 0 And strUseProfile <> "CUSTOM") Or (strUseProfile = "LEVEL") Then
   
                        ' // Variable to store the event probability (default to 1)
                        dblActivityProbability = 1
                        
                        ' // For appliances that depend on activity profiles and is not a custom profile ...
                        If (strUseProfile <> "LEVEL") And (strUseProfile <> "ACTIVE_OCC") And (strUseProfile <> "CUSTOM") Then
                        
                            ' // Get the activity statistics for this profile
                            Set objActivityStatsItem = objActivityStatistics.Item(strKey)
    
                            ' // Get the probability for this activity profile for this time step
                            dblActivityProbability = objActivityStatistics(strKey).Modifiers(intTenMinuteCount)
                                                
                        End If
    
                        ' // Check the probability of the activity occurring and the appliance switching on
                        If (Rnd() < (dblActivityProbability * dblProbSwitchOn)) Then
                        
                            ' // This is a start event
                            StartAppliance
                            
                        End If
         
                    End If
                Else
                    ' // The appliance is on - if the occupants become inactive, switch off the appliance
                    If (intActiveOccupants = 0) And (strUseProfile <> "LEVEL") And (strUseProfile <> "ACT_LAUNDRY") And (strUseProfile <> "CUSTOM") Then
                            
                        ' // Do nothing. The activity will be completed upon the return of the active occupancy.
                        ' // Note that LEVEL means that the appliance use is not related to active occupancy.
                        ' // Note also that laundry appliances do not switch off upon a transition to inactive occupancy.
                    Else
      
                        ' // Set the power
                        intPower = GetPowerUsage(intCycleTimeLeft)
                        
                        ' // Decrement the cycle time left
                        intCycleTimeLeft = intCycleTimeLeft - 1
                        
                    End If
                End If
    
                ' // Set the appliance power at this time step
                aSimulationArray(2 + intMinute, intAppliance) = intPower
            
                ' // Increment the time
                intMinute = intMinute + 1
            Loop
            
        End If
        'wsMain.Range("L16") = "Appliance number " + CStr(intAppliance) + " Completed"
        'Stop
    Next intAppliance
    
End Sub




' ===============================================================================================
' ## TotalApplianceDemand
' Calculate the sum of the appliance demand (including heating system demand)
' ===============================================================================================
Public Sub TotalApplianceDemand()
    Dim intRow As Integer
    Dim intCol As Integer
    Dim dblRowSum As Double
    
    For intRow = 1 To 1440
    
        dblRowSum = 0
        
        For intCol = 1 To 31
            
            dblRowSum = dblRowSum + aSimulationArray(intRow + 2, intCol)
            
        Next intCol
        
        aTotalApplianceDemand(intRow, 1) = dblRowSum _
            + aPrimaryHeatingSystem(intRunNumber).GetHeatingSystemPowerDemand(intRow) _
            + aSolarThermal(intRunNumber).GetP_pumpsolar(intRow) _
            + aCoolingSystem(intRunNumber).GetCoolingSystemPowerDemand(intRow)
        
    Next intRow
    
End Sub




' ===============================================================================================
' ## WriteAppliances
' Write the simulation results to the worksheet
' ===============================================================================================
Public Sub WriteAppliances(dwellingIndexRowOffset As Long)

    Dim intOffset As Integer
    Dim intAppliance As Integer
    
    intOffset = 6
        
    With wsResultsDisaggregated.Range("G" + CStr(1 + intOffset) + ":G" + CStr(1440 + intOffset))
        .Offset(dwellingIndexRowOffset, 0) = aTotalApplianceDemand
    End With
    
    With wsDwellings
        For intAppliance = 1 To 31
        .Cells(intDwellingIndex + 4, intAppliance + 8).Value = aApplianceConfiguration(intAppliance, 1) * -1
        Next intAppliance
    End With
    
End Sub




' ===============================================================================================
' ## CalculateThermalGains
' Calculate thermal gains associated with the appliances
' ===============================================================================================
Public Sub CalculateThermalGains()

    Dim intMinute As Integer
    Dim intApplianceIndex As Integer
    Dim dblSum As Double
    Dim vntHeatGainsRatio() As Variant
    Dim dblAppliancePower As Double
        
    vntHeatGainsRatio = wsAppliancesAndWaterFixtures.Range("rHeatGainsRatio").Value
    
    For intMinute = 1 To 1440
        
        dblSum = 0
        
        For intApplianceIndex = 1 To 31
            If aApplianceConfiguration(intApplianceIndex, 1) = True Then
                
                dblAppliancePower = aSimulationArray(intMinute + 2, intApplianceIndex)
            
                dblSum = dblSum + dblAppliancePower * vntHeatGainsRatio(intApplianceIndex, 1)
                
            End If
        Next intApplianceIndex
        
        aApplianceThermalGains(intMinute, 1) = dblSum
        
    Next intMinute
    
End Sub




' ===============================================================================================
' ## StartAppliance
' Start a cycle for the current appliance
' ===============================================================================================
Private Sub StartAppliance()

    ' // Determine how long this appliance is going to be on for
    intCycleTimeLeft = CycleLength()
    
    ' // Determine if this appliance has a delay after the cycle before it can restart
    intRestartDelayTimeLeft = intRestartDelay
    
    ' // Set the power
    intPower = GetPowerUsage(intCycleTimeLeft)
    
    ' // Decrement the cycle time left
    intCycleTimeLeft = intCycleTimeLeft - 1
                            
End Sub




' ===============================================================================================
' Class Functions

' ===============================================================================================
' ## CycleLength
' Start a cycle for the current appliance
' ===============================================================================================
Private Function CycleLength() As Integer
    
    ' // Set the value to that provided in the configuration
    CycleLength = intMeanCycleLength
    
    ' // Use the TV watching length data approximation, derived from the TUS data
    If (strApplianceType = "TV1") Or (strApplianceType = "TV2") Or (strApplianceType = "TV3") Then
    
        ' // The cycle length is approximated by the following function
        ' // The avergage viewing time is approximately 73 minutes
        CycleLength = CInt(70 * ((0 - Log(1 - Rnd())) ^ 1.1))
        
    ElseIf (strApplianceType = "STORAGE_HEATER") Or (strApplianceType = "ELEC_SPACE_HEATING") Then
    
        ' // Provide some variation on the cycle length of heating appliances
        CycleLength = GetMonteCarloNormalDistGuess(CDbl(intMeanCycleLength), intMeanCycleLength / 10)
    
    End If
        
End Function




' ===============================================================================================
' ## GetPowerUsage
'
' ===============================================================================================
Private Function GetPowerUsage(ByVal intCycleTimeLeft As Integer) As Integer

    ' // Set the return power to the rated power
    GetPowerUsage = intRatedPower

    ' // Some appliances have a custom (variable) power profile depending on the time left
    Select Case strApplianceType
    
        Case "WASHING_MACHINE", "WASHER_DRYER":
        
            ' // Declare the total cycle time variable
            Dim iTotalCycleTime As Integer
        
            ' // Calculate the washing cycle time
            If (strApplianceType = "WASHING_MACHINE") Then iTotalCycleTime = 138
            If (strApplianceType = "WASHER_DRYER") Then iTotalCycleTime = 198
        
            ' // This is an example power profile for an example washing machine
            ' // This simplistic model is based upon data from personal communication with a major washing maching manufacturer
            Select Case (iTotalCycleTime - intCycleTimeLeft + 1)
            
                Case 1 To 8: GetPowerUsage = 73         ' // Start-up and fill
                Case 9 To 29: GetPowerUsage = 2056     ' // Heating
                Case 30 To 81: GetPowerUsage = 73       ' // Wash and drain
                Case 82 To 92: GetPowerUsage = 73       ' // Spin
                Case 93 To 94: GetPowerUsage = 250      ' // Rinse
                Case 95 To 105: GetPowerUsage = 73      ' // Spin
                Case 106 To 107: GetPowerUsage = 250    ' // Rinse
                Case 108 To 118: GetPowerUsage = 73     ' // Spin
                Case 119 To 120: GetPowerUsage = 250    ' // Rinse
                Case 121 To 131: GetPowerUsage = 73     ' // Spin
                Case 132 To 133: GetPowerUsage = 250    ' // Rinse
                Case 134 To 138: GetPowerUsage = 568    ' // Fast spin
                Case 139 To 198: GetPowerUsage = 2500   ' // Drying cycle
                Case Else: GetPowerUsage = intStandbyPower
            
            End Select
                            
    End Select

End Function
