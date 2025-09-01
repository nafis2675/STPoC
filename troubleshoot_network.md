# Network Access Troubleshooting Guide

## 🔍 Problem: Can access from this PC but not from other PC

### Current Status:
- ✅ Server is running (restarted in background)
- ✅ Windows Firewall allows Python
- ❌ Other PC cannot connect

### Most Common Causes & Solutions:

## 1. 🌐 **Network Connection Issues**

### Check if both computers are on the same network:
- **This PC**: Connected to WiFi/Ethernet
- **Other PC**: Must be on **the same WiFi network**
- **Corporate Network**: May have device isolation enabled

### Solution:
```bash
# On this PC - check your IP:
ipconfig | findstr "IPv4"

# On other PC - try to ping this PC:
ping [YOUR_IP_ADDRESS]
```

## 2. 🔒 **Corporate/Domain Network Restrictions**

**Your PC is on a domain network** (`kanto.sysystem`), which often blocks:
- Inter-device communication
- Custom applications
- Port 8000 access

### Solutions:
1. **Use localhost on same PC**: `http://localhost:8000`
2. **Ask IT Admin**: Request port 8000 access
3. **Use different port**: Try port 3000, 5000, or 9000
4. **VPN/Hotspot**: Create phone hotspot, connect both PCs

## 3. 🛡️ **Windows Network Discovery**

### Enable Network Discovery:
1. Control Panel → Network and Sharing Center
2. Change advanced sharing settings
3. Turn ON:
   - Network discovery
   - File and printer sharing
   - Public folder sharing (for Public networks)

## 4. 🔌 **Change Server Port**

Corporate networks often block port 8000. Try different ports:

```python
# In app.py, change the last line to:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)  # Try 3000, 5000, 9000
```

Then access via: `http://[YOUR_IP]:3000`

## 5. 🔄 **Alternative Solutions**

### Option A: Use Phone Hotspot
1. Create hotspot on phone
2. Connect both PCs to phone's WiFi
3. Access app normally

### Option B: Same PC Multiple Browsers
1. Use Chrome on same PC: `http://localhost:8000`
2. Use Edge on same PC: `http://127.0.0.1:8000`
3. Use Firefox on same PC: `http://[YOUR_IP]:8000`

### Option C: Test with Simple Server
```bash
# Test if network works with simple Python server:
python -m http.server 8080
# Then try accessing from other PC: http://[YOUR_IP]:8080
```

## 6. 🧪 **Diagnostic Steps**

### From Other PC, test these URLs:
1. `http://[THIS_PC_IP]:8000` - Main app
2. `http://[THIS_PC_IP]:8000/health` - Health check
3. `ping [THIS_PC_IP]` - Basic connectivity

### Check Windows Event Viewer:
1. Windows key + R → `eventvwr`
2. Windows Logs → System
3. Look for network/firewall blocks

## 7. 🔧 **Quick Fixes to Try**

1. **Restart both computers**
2. **Disable Windows Firewall temporarily** (just for testing)
3. **Use incognito/private browser** on other PC
4. **Try different browser** (Chrome, Edge, Firefox)
5. **Check if other PC has proxy** settings

## 8. 📱 **Mobile Testing**

Instead of other PC, try from your phone:
1. Connect phone to same WiFi
2. Go to: `http://[YOUR_IP]:8000`
3. If mobile works but PC doesn't = PC-specific issue

---

**Most Likely Solution for Corporate Network**: 
Use phone hotspot or ask IT to allow port 8000 access between devices on the domain network.
