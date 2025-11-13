Attribute VB_Name = "clsHotWater"
Attribute VB_Base = "0{FCFB3D2A-A0FA-1068-A738-08002B3371B5}"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Attribute VB_TemplateDerived = False
Attribute VB_Customizable = False
' ===============================================================================================
' ## Hot Water Demand Model
' ===============================================================================================


Option Explicit

' ===============================================================================================
' Class Variables

' // Declare variables

Private intDwellingIndex As Integer
Private intRunNumber As Integer

Private intResidents As Integer
Private intHeatingSystemIndex As Integer

' litres per day per dwelling
Private dblHotWaterBaseDemand As Double

Private aHourlyHotWaterProfile(0 To 25) As Double

' An array to store the hot water demand in litres per minute.
Private aHotWaterDemand(1 To 1440, 1 To 1) As Double

Private aActiveOccupancy() As Integer

' an array to store the temperature of the hot water in the hot water cylinder
Private aTheta_cyl(1 To 1440, 1 To 1) As Double

' thermal capacitance of hot water cylinder
Private dblC_cyl As Double

' temperature of cold water inlet
Private dblTheta_cw As Double

' thermal transfer coefficient between hot water tank and dwelling internal building node
Private dblH_loss As Double

' array to store the variable thermal heat transfer coefficient associated with the hot water demand
Private aH_demand(1 To 1440, 1 To 1) As Double

' dwelling water fixture configuration
Private aWaterFixtureConfiguration(1 To 4, 1 To 1) As Boolean

' a variable to store the time left on for a water flow event
Private dblEventTimeLeft As Double
Private dblRestartTimeLeft As Double

' a variable to store the flow rate for a water draw event
Private dblFlow As Double

' // declare the fixture description variables
Private strFixtureName As String
Private dblMeanEventDuration As Double
Private dblEventsPerYear As Double
Private dblMeanFlow As Double
Private dblProbSwitchOn As Double
Private dblOwnership As Double
Private dblTargetAverageVolumeYear As Double
Private strUseProfile As String
Private blnHasFixture As Boolean
Private intEventFlow As Integer
Private dblRestartDelay As Double

' an array to store the fixture hot water draws for the simulation
Private aSimulationArray(1 To 1442, 1 To 4) As Variant




' ===============================================================================================
' Class Properties

'Public Property Get GetTheta_cyl(timestep As Integer) As Double
'    GetTheta_cyl = aTheta_cyl(timestep, 1)
'End Property

Public Property Get GetC_cyl() As Double
    GetC_cyl = dblC_cyl
End Property

Public Property Get GetTheta_cw() As Double
    GetTheta_cw = dblTheta_cw
End Property
Public Property Get GetH_demand(currentTimeStep As Integer) As Double
    GetH_demand = aH_demand(currentTimeStep, 1)
End Property

Public Property Get GetH_loss() As Double
    GetH_loss = dblH_loss
End Property

Public Property Get GetDailySumHotWaterDemand() As Double
    GetDailySumHotWaterDemand = WorksheetFunction.Sum(aHotWaterDemand)
End Property

' ===============================================================================================
' Class Subroutines

' ===============================================================================================
' ## InitialiseHotWater
'
' ===============================================================================================
Public Sub InitialiseHotWater(dwellingIndex As Integer, runNumber As Integer)
    
    Dim intHour As Integer
    Dim intOffset As Integer
    
    Dim intRow As Integer
    
    Dim dblRand As Double
    
    Dim dblProportion As Double
    
    ' DHW cylinder volume
    Dim dblV_cyl As Double
    
    intOffset = 4
    
    ' // Get dwelling's hot water variables
    intDwellingIndex = dwellingIndex
    intRunNumber = runNumber
    With wsDwellings
        intResidents = Application.index(.Range("rNumberResidentsIndex"), intOffset + intDwellingIndex).Value
        intHeatingSystemIndex = Application.index(.Range("rPrimaryHeatingIndex"), intOffset + intDwellingIndex).Value
    End With
        
    ' // Get the active occupancy array for this dwelling
    aActiveOccupancy = aOccupancy(intRunNumber).GetActiveOccupancy
    
    ' // Get hot water system parameters
    With wsPrimaryHeatingSystems
        
        ' // Get thermal resistance between hot water tank and internal building node
        dblH_loss = Application.index(.Range("rH_loss"), intOffset + intHeatingSystemIndex).Value
                
        ' // Calculate thermal capacity of hot water tank
        dblV_cyl = Application.index(.Range("rV_cyl"), intOffset + intHeatingSystemIndex).Value
        dblC_cyl = SPECIFIC_HEAT_CAPACITY_WATER * dblV_cyl
        
    End With
    
    
    ' // Set cold water inlet temperature
    If blnUK Then
        dblTheta_cw = 10
    ElseIf blnIndia Then
        dblTheta_cw = 20
    Else
        Stop 'country not recognised yet
    End If
                
    ' // Configure the water fixtures in the dwelling
    
    ' // For each fixture
    For intRow = 1 To 4
        
        ' // Pick a random number
        dblRand = Rnd()
        
        ' // Get the proportion of dwellings with this fixture
        dblProportion = wsAppliancesAndWaterFixtures.Range("rWaterFixtureProportion").Cells(intRow, 1).Value
        
        ' // Determine if this dwelling has this fixture
        aWaterFixtureConfiguration(intRow, 1) = IIf(dblRand < dblProportion, True, False)
        
    Next intRow
    
End Sub



' ===============================================================================================
' ## RunHotWaterDemandSimulation
'
' ===============================================================================================
Public Sub RunHotWaterDemandSimulation()

    ' // Declare variables
    Dim intFixture As Integer
    Dim blnWeekend As Boolean
    Dim intMinute As Integer
    Dim intTenMinuteCount As Integer
    Dim intActiveOccupants As Integer
    Dim dblRand As Double
    Dim dblActivityProbability As Double
    Dim strKey As String
    Dim strCell As String
    Dim LastCell As Range
    
    ' cell offset
    Dim intRowOffset As Integer
    
    intRowOffset = 45
        
    ' // For each of the fixtures
    For intFixture = 1 To 4
        
        ' // Initialisation
        dblEventTimeLeft = 0
        
        ' // Get the water fixture details
        With wsAppliancesAndWaterFixtures
            strFixtureName = .Range("E" + CStr(intFixture + intRowOffset)).Value
            dblProbSwitchOn = .Range("AD" + CStr(intFixture + intRowOffset)).Value
            dblMeanFlow = .Range("P" + CStr(intFixture + intRowOffset)).Value
            strUseProfile = .Range("G" + CStr(intFixture + intRowOffset)).Value
            dblRestartDelay = .Range("S" + CStr(intFixture + intRowOffset)).Value
        End With
        blnHasFixture = aWaterFixtureConfiguration(intFixture, 1)
        
        ' // Write fixture type to the simulation array
        aSimulationArray(1, intFixture) = strFixtureName
        
        ' // Write the units
        aSimulationArray(2, intFixture) = "(litres/min)"
        
        ' // Check if the dwelling has this fixture
        If (Not blnHasFixture) Then
        
            ' // This fixture is not applicable to write zeros to the water demand
            Dim intCount As Integer
            For intCount = 3 To 1442
                aSimulationArray(intCount, intFixture) = 0
            Next intCount
            
        Else
                        
            ' // Get the weekend or weekday flag
            strCell = wsMain.Range("rDayType").Value
            blnWeekend = IIf(strCell = "we", True, False)
            
            ' // Loop through each minute of the day
            intMinute = 1
            
            Do While (intMinute <= 1440)
                
                ' // Set the default flow rate of the fixture (zero)
                dblFlow = 0
                
                ' // Get the ten minute period count
                intTenMinuteCount = ((intMinute - 1) \ 10)
                
                ' // Get the number of current active occupants for this minute
                intActiveOccupants = aActiveOccupancy(intTenMinuteCount, 0)
                
                ' // Now generate a key to get the activity statistics for this fixture
                strKey = IIf(blnWeekend, "1", "0") + "_" + CStr(intActiveOccupants) + "_" + strUseProfile
                
                ' // If this fixture is off but a restart needs to be delayed
                If (dblEventTimeLeft <= 0) And (dblRestartTimeLeft > 0) Then
                    
                    ' // Decrement the restart time left
                    
                    dblRestartTimeLeft = dblRestartTimeLeft - 1
                    
                ' // Else if this fixture is not being used, or a restart has not been delayed
                ElseIf dblEventTimeLeft <= 0 Then
                    
                    ' // There needs to be active occupants for a start event to occur
                    If intActiveOccupants > 0 Then
                                                
                        ' // Get the activity statistics for this profile
                        Set objActivityStatsItem = objActivityStatistics.Item(strKey)
                        
                        ' // Get the probability for this activity profile for this time step
                        dblActivityProbability = objActivityStatistics(strKey).Modifiers(intTenMinuteCount)
                        
                        ' // Check the probability of this activity occurring and there being a start event
                        If (Rnd() < (dblActivityProbability * dblProbSwitchOn)) Then
                            
                            ' // This is a start event
                            StartFixture intFixture, intMinute
                            
                        End If
                    End If
                Else
                    ' // The fixture is on - if the occupancts become inactive, then turn off the fixture
                    If intActiveOccupants = 0 Then
                    
                        ' // Do nothing. The activity will be completed upon the return of the active occupants
                    Else
                        ' // Assign the power flow
                        
                        ' // If the amount of time left to run the fixture is less than a minute, proportionally set the flow rate
                        If dblEventTimeLeft < 1 Then
                            dblFlow = dblMeanFlow * dblEventTimeLeft
                        Else
                            ' // Otherwise the flow rate is the mean flow rate
                            dblFlow = dblMeanFlow
                        End If
                        
                        ' // Decrement the event time left
                        dblEventTimeLeft = dblEventTimeLeft - 1
                    End If
                End If
                
                ' // Set the fixture water flow at this time step
                aSimulationArray(2 + intMinute, intFixture) = dblFlow
                
                ' // increment the time
                intMinute = intMinute + 1
            Loop
        End If
        
    Next intFixture
    
    ' // Calculate the sum for all the water fixtures and the resulting thermal transfer coefficient
    TotalHotWaterDemandAndThermalTransferCoefficient
    
End Sub




' ===============================================================================================
' ## StartFixture
'
' ===============================================================================================
Private Sub StartFixture(intFixture As Integer, intMinute As Integer)
    Dim dblRand As Double
    Dim dblCumulativeP As Double
    Dim intRow As Integer
    Dim intCol As Integer
    Dim dblEventVolume As Double
    
    ' // Assign the fixture's restart delay (can be zero)
    dblRestartTimeLeft = dblRestartDelay
    
    ' // Assign the flow rate to be the mean flow rate for this type of fixture
    dblFlow = dblMeanFlow
    
    ' // Now stochastically determine the volume for this fixture event
    
    ' // Determine the column for the relevant probability distribution
    Select Case intFixture
        ' // Basins and sinks have the same event volume probability distribution
        Case 1, 2
            intCol = 3
        ' // Showers
        Case 3
            intCol = 4
        ' // Baths
        Case 4
            intCol = 5
    End Select
    
    ' // Pick a random number
    dblRand = Rnd()
    
    ' // Set the cumulative probability to zero
    dblCumulativeP = 0
    
    ' // iterate through the probability distribution
    For intRow = 1 To 151
        
        ' // Add the probability
        dblCumulativeP = dblCumulativeP + wsWaterUsage.Range("rWaterUsageStatistics").Cells(intRow, intCol).Value
        
        If dblRand < dblCumulativeP Then
            ' // This determines the fixture event volume
            dblEventVolume = wsWaterUsage.Range("rWaterUsageStatistics").Cells(intRow, 1).Value
            
            Exit For
        End If
    Next intRow
    
    ' // the event volume is determined to be zero then set the flow rate to zero
    If dblEventVolume = 0 Then
        dblFlow = 0
        dblEventTimeLeft = 0
    Else
        ' // Otherwise calculate the event duration
        dblEventTimeLeft = dblEventVolume / dblMeanFlow
        
        ' // If the event duration is a fraction of a minute, then proportionally allocate the flow rate
        If dblEventTimeLeft < 1 Then
            dblFlow = dblMeanFlow * dblEventTimeLeft
        End If
            
        ' // and decrement the event time left
        dblEventTimeLeft = dblEventTimeLeft - 1
    End If
    
    
    
End Sub




' ===============================================================================================
' ## TotalHotWaterDemandAndThermalTransferCoefficient()
' Calculate the sum of hot water demand from each fixture for each time step
' ===============================================================================================
Public Sub TotalHotWaterDemandAndThermalTransferCoefficient()
    Dim intMinute As Integer
    Dim intFixture As Integer
    Dim dblRowSum As Double
    
    ' hot water mass flow rate (kg/s)
    Dim dblM_w As Double
    
     ' volume flow rate in m^3 per second
    Dim dblV_w As Double
    
    ' density of water in kg per m^3
    Dim dblRho_w As Double
    
    ' // set the density of water
    dblRho_w = 1000
    
    ' // for each minute
    For intMinute = 1 To 1440
        
        ' // sum the hot water demand from all the fixtures
        dblRowSum = 0
        
        For intFixture = 1 To 4
        
            dblRowSum = dblRowSum + aSimulationArray(intMinute + 2, intFixture)
        
        Next intFixture
        
        aHotWaterDemand(intMinute, 1) = dblRowSum
        
        ' // Then calculate variable thermal resistance representing hot water demand
        
        ' // conversion from litres to m^3 and from per minute to per second
        dblV_w = aHotWaterDemand(intMinute, 1) / 1000 / 60
        
        ' // to convert from m^3 per second to kg per second
        dblM_w = dblRho_w * dblV_w
        
        ' // convert to a thermal heat transfer coefficient in W/K
        aH_demand(intMinute, 1) = SPECIFIC_HEAT_CAPACITY_WATER * dblM_w
        
    Next intMinute
    
End Sub




' ===============================================================================================
' ## WriteHotWater
'
' ===============================================================================================
Public Sub WriteHotWater(dwellingIndexRowOffset As Long)

    With wsResultsDisaggregated
        .Range("O7:O1446").Offset(dwellingIndexRowOffset, 0) = aHotWaterDemand
    End With
    
End Sub


